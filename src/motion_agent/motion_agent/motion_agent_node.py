#!/usr/bin/env python3
"""
motion_agent_node.py  (Step FULL PICK-AND-PLACE SEQUENCE)
==============================================================
On a 'pick' command it runs the complete sequence (blocking, step by step):

  1. OPEN gripper        (start open)
  2. PRE-GRASP           (hover above object, pointing down)
  3. DESCEND             (lower onto object)
  4. CLOSE gripper       (grab)
  5. LIFT                (raise with object)
  6. CARRY to bin        (pose above correct bin: bottle->left, can->right)
  7. OPEN gripper        (release into bin)
  8. HOME                (return to ready joint pose)

Execution path: this node -> /move_action (MoveIt) -> trajectory_bridge
-> /joint_command -> Isaac. Gripper -> /joint_command directly.

Run the bridge in AUTO mode (auto_execute:=true) so steps run without ENTER.
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from robot_interfaces.msg import LLMCommand, VisionResult
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, JointConstraint,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# --- Metrics logging (optional, removable). Safe no-op if unavailable. ---
try:
    from motion_agent.metrics_logger import log_event
except Exception:
    def log_event(*args, **kwargs):   # no-op fallback
        pass


class MotionAgentNode(Node):
    def __init__(self):
        super().__init__('motion_agent_node')

        # ---- Frame / arm config ----
        # Robot base position in Isaac world coords (metres). Must match the
        # arm's placement in the simulation; world_to_base() uses it to convert
        # perception's world coordinates into the arm's base_link frame.
        self.robot_base_world = (0.0, -0.718, 2.43)
        self.planning_group = 'arm'
        self.planning_frame = 'base_link'
        self.eef_link = 'link6'

        # ---- Grasp geometry (all metres) ----
        self.gripper_length = 0.136     # link6 -> fingertip
        self.tilt_offset_x = 0.070      # compensate gripper tilt (forward)
        self.tilt_offset_y = 0.027


        # ---- STANDING SIDE-GRASP (human-like) ----
        # Captured empirically via test_joint [0.0, 2.4, -1.4, 0.0, -0.7, 0.0]
        # and tf2_echo base_link -> link6:
        #   link6 quat (xyzw) = (-0.005, 0.779, -0.001, 0.627)
        #   tool axis points ~+X with ~12 deg downward tilt
        #   object sat ~0.133m ahead of link6, at height ~0.049m (can mid-body)
        # For an object at yaw angle theta, both the orientation and the
        # approach direction are rotated by theta around Z, so joint1 turns
        # automatically for side objects via IK.
        self.side_quat_template = (-0.005, 0.779, -0.001, 0.627)
        self.side_back_offset = 0.100      # link6 sits this far behind object
        self.side_z_offset = 0.032         # link6 above fingertip height
        self.side_approach_extra = 0.07    # pre-approach: extra distance back
        # NOTE: the side grasp needs the object at arm's length. Template was
        # captured with the object ~0.63m from the base. Objects closer than
        # ~0.45m pull the link6 target inside minimum reach -> IK failure.
        # Single side-grasp height (fingertip above table point). Objects are
        # similar in height, so one value is used. Lowered slightly from the
        # previous 0.057 so the fingers sit a little further down the body.
        self.standing_side_grasp_height = 0.050


        # ---- LYING-object grasp config (separate branch) ----
        # Wrist rotation so fingers close ACROSS the round body.
        # Found empirically: +90 deg from standing (joint6 = 0.0).
        self.lying_wrist_angle = 1.5708   # radians (joint6 for lying)
        # Extra hover height for the lying pre-grasp (before twisting wrist)
        self.lying_pregrasp_height = 0.09        # metres above object center

        # ----lying grasp (twist high, then joint-space descend) ----
        # Height above object center at which we TWIST (high enough that the
        # turning fingers clear a fat body like the big bottle). Tune up if the
        # big bottle still gets knocked; tune down if the twist is too high.
        self.lying_twist_clearance = 0.055      # metres above object center
        # After twisting, we lower the hand by bending joints 2 & 3 a little.
        # These are the per-step joint deltas that move the hand ~straight down.
        # Calibrate empirically (like bin poses): increase to drop more.
        self.lying_descend_j2 = 0.10            # radians added to joint2
        self.lying_descend_j3 = 0.10          # radians added to joint3
        # Safety: max descend steps (each step applies the deltas once).
        self.lying_descend_steps = 1

        # Measured Piper "gripper pointing down" orientation (x,y,z,w)
        self.down_quat = (0.004, 0.976, -0.001, 0.217)


        # Home joint pose (joint1..joint6)
        self.home_joints = [0.0, 0.8, -1.0, 0.0, 1.0, 0.0]
        self.raise_joints = [0.0, 1.2, -1.2, 0.0, 1.2, 0.0]
        # Fixed bin poses (joint space) — no IK, no twist
        # joint1 = +1.5 -> right bin (cans), -1.5 -> left bin (bottles)
        self.bin_poses = {
            'can': {     # right bin
                'above': [1.5, 1.45, -1.75, 0.0, 1.2, 0.0],
                'drop':  [1.5, 1.9, -1.6,  0.0, 1.15, 0.0],
            },
            'bottle': {  # left bin
                'above': [-1.5, 1.45, -1.75, 0.0, 1.2, 0.0],
                'drop':  [-1.5, 1.9, -1.6,  0.0, 1.15, 0.0],
            },
        }

        # Gripper open/close joint values
        self.grip_open = [0.05, -0.05]    # joint7, joint8
        self.grip_close = [0.0, 0.0]

        # ---- State ----
        self.latest_vision = None
        self._result_code = None
        self._result_event = threading.Event()
        self._latest_joint_state = None   # for lying wrist override
        self._busy = False                # True while a pick sequence runs
        self._busy_lock = threading.Lock()
        self._seq_ok = False              # set True when a pick completes fully

        # ---- Subscribers ----
        self.create_subscription(
            LLMCommand, '/llm/command', self.command_callback, 10)
        self.create_subscription(
            VisionResult, '/perception/result', self.vision_callback, 10)
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        # ---- MoveIt action client ----
        self.move_client = ActionClient(self, MoveGroup, '/move_action')

        # ---- Gripper publisher ----
        self.gripper_pub = self.create_publisher(
            JointState, '/joint_command', 10)

        self.get_logger().info('Motion Agent started.')

    # ==================================================================
    def vision_callback(self, msg):
        self.latest_vision = msg

    # ==================================================================
    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg

    def _read_arm_joints(self, timeout=3.0):
        """Return current [joint1..joint6] from /joint_states, or None."""
        names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        deadline = time.time() + timeout
        while time.time() < deadline:
            js = self._latest_joint_state
            if js is not None and js.name:
                lookup = dict(zip(js.name, js.position))
                if all(n in lookup for n in names):
                    return [float(lookup[n]) for n in names]
            time.sleep(0.05)
        return None

    # ==================================================================
    def command_callback(self, msg):
        print('\n' + '=' * 60)
        print(f'  LLM COMMAND: intent={msg.intent} target={msg.target_object}')
        print('-' * 60)
        log_event('motion', 'command_received',
                  trial_id=msg.target_object, target=msg.target_object)

        # --- simple test intents kept for debugging ---
        if msg.intent == 'gripper_open':
            self.set_gripper(True);  return
        if msg.intent == 'gripper_close':
            self.set_gripper(False); return
        if msg.intent == 'test_joint':
            # Debug intent only. The six joint angles are passed as a
            # comma-separated string in the 'reasoning' field (reused here as a
            # data channel), e.g. reasoning="0.0,2.4,-1.4,0.0,-0.7,0.0".
            try:
                vals = [float(x) for x in msg.reasoning.split(',')]
                if len(vals) == 6:
                    print(f'  TEST JOINT: {vals}')
                    print('=' * 60 + '\n')
                    threading.Thread(
                        target=self._goto_joints, args=(vals,), daemon=True
                    ).start()
                    return
            except Exception as e:
                print(f'  bad test_joint values: {e}')
            return

        # BULK CLEAR: sort many objects in one command, nearest-first,
        # optionally filtered by class and/or pose. Runs its own guarded loop.
        if msg.intent == 'clear_all':
            fclass = (getattr(msg, 'filter_class', '') or '').strip().lower()
            fpose = (getattr(msg, 'filter_pose', '') or '').strip().lower()
            with self._busy_lock:
                if self._busy:
                    print('  [!] Busy with a pick sequence — ignoring clear_all '
                          'until it finishes.')
                    print('=' * 60 + '\n')
                    return
                self._busy = True
            threading.Thread(
                target=self._run_clear_all_guarded, args=(fclass, fpose),
                daemon=True
            ).start()
            return

        if msg.intent != 'pick':
            print(f'  [i] intent "{msg.intent}" not handled in 5e.')
            print('=' * 60 + '\n')
            return

        target = self._find_target(msg.target_object)
        if target is None:
            print(f'  [!] Target "{msg.target_object}" not found.')
            print('=' * 60 + '\n')
            return

        # BUSY GUARD (bug fix): if a pick sequence is already running, ignore
        # the new command instead of starting a second, concurrent sequence.
        # Two concurrent sequences send conflicting goals and make the arm jump
        # from the bin straight to the next object, dragging through the scene.
        # One pick at a time; new commands are accepted only after rise + home.
        with self._busy_lock:
            if self._busy:
                print('  [!] Busy with a pick sequence — ignoring new command '
                      'until it finishes.')
                print('=' * 60 + '\n')
                return
            self._busy = True

        # Run the whole sequence on a separate thread so callbacks keep
        # flowing while we block waiting for each motion result.
        threading.Thread(
            target=self._run_pick_sequence_guarded, args=(target,),
            daemon=True
        ).start()

    def _run_pick_sequence_guarded(self, target):
        """Wrapper: always clears the busy flag when the sequence ends,
        even if it aborts, so the node can accept the next command."""
        try:
            self._run_pick_sequence(target)
        finally:
            with self._busy_lock:
                self._busy = False
            print('  [i] Ready for next command.\n')

    # ==================================================================
    def _pickable_objects(self, fclass, fpose):
        """Return current pickable objects from the latest perception result,
        filtered by class and pose (empty filter = any), sorted NEAREST first.
        Re-reads self.latest_vision each call, so it reflects the live scene.
        """
        if self.latest_vision is None:
            return []
        out = []
        for obj in self.latest_vision.objects:
            # class filter
            if fclass and obj.object_class.strip().lower() != fclass:
                continue
            # pose filter
            if fpose and obj.pose.strip().lower() != fpose:
                continue
            # only reachable/pickable objects
            if hasattr(obj, 'reachable') and not obj.reachable:
                continue
            out.append(obj)
        # nearest first (perception provides distance_from_robot)
        out.sort(key=lambda o: float(getattr(o, 'distance_from_robot', 0.0)))
        return out

    def _run_clear_all_guarded(self, fclass, fpose):
        """Bulk-clear loop: repeatedly pick the nearest matching object until
        none remain. Re-scans perception before each pick. Always clears the
        busy flag at the end so the node can accept new commands.
        """
        filt = []
        if fclass:
            filt.append(f'class={fclass}')
        if fpose:
            filt.append(f'pose={fpose}')
        filt_str = (' [' + ', '.join(filt) + ']') if filt else ' [everything]'
        print('\n' + '=' * 60)
        print(f'  CLEAR ALL{filt_str} — starting bulk sort')
        print('=' * 60)
        try:
            done = 0
            failed_labels = set()   # objects that failed; skip on re-scan
            # Safety cap so detection flicker can't loop forever.
            max_picks = 10
            while done < max_picks:
                # small settle so perception reflects the just-dropped object
                time.sleep(1.0)
                pending = self._pickable_objects(fclass, fpose)
                # drop objects we already failed to pick this run
                pending = [o for o in pending
                           if o.label.strip().lower() not in failed_labels]
                if not pending:
                    print(f'\n  >>> CLEAR ALL COMPLETE — sorted {done} object(s).')
                    if failed_labels:
                        print(f'      Skipped {len(failed_labels)} unreachable/'
                              f'failed: {sorted(failed_labels)}')
                    print('=' * 60 + '\n')
                    return
                target = pending[0]      # nearest matching object
                remaining = len(pending)
                print(f'\n  [clear_all] {remaining} left; picking nearest: '
                      f'"{target.label}" '
                      f'(dist={float(getattr(target, "distance_from_robot", 0.0)):.3f})')
                # Reuse the SAME single-object sequence (standing/lying auto).
                ok = self._run_pick_sequence(target)
                if ok:
                    done += 1
                else:
                    # Skip this object next time so we don't retry it forever.
                    failed_labels.add(target.label.strip().lower())
                    print(f'  [clear_all] "{target.label}" failed — skipping it '
                          f'and moving on.')
            print(f'\n  [!] clear_all hit the safety cap ({max_picks}); stopping.')
            print('=' * 60 + '\n')
        finally:
            with self._busy_lock:
                self._busy = False
            print('  [i] Ready for next command.\n')

    # ==================================================================
    def _run_pick_sequence(self, target):
        """Dispatcher: reads pose and routes to standing or lying.
        Returns True if the sequence ran to completion, False if it aborted.
        """
        label = target.label
        obj_class = target.object_class.strip().lower()
        pose = target.pose.strip().lower()   # 'standing' or 'lying'
        bx, by, bz = self.world_to_base(
            target.world_x, target.world_y, target.world_z)
        print(f'  Target "{label}" class={obj_class} ({target.pose})')
        print(f'    object base: ({bx:.3f}, {by:.3f}, {bz:.3f})')
        log_event('motion', 'sequence_start',
                  trial_id=label, target=label, pose=pose)

        # Completion flag: the pick functions set this True at their end.
        self._seq_ok = False
        if pose == 'lying':
            self._run_lying_pick(target, obj_class, bx, by, bz)
        else:
            # Standing objects use the human-like SIDE grasp.
            self._run_standing_side_pick(target, obj_class, bx, by, bz)
        log_event('motion', 'sequence_end',
                  trial_id=label, target=label, pose=pose)
        return self._seq_ok

    @staticmethod
    def _yaw_rotated_quat(theta, q):
        """Rotate quaternion q (x,y,z,w) by theta around the base Z axis.
        Returns q_yaw (Hamilton product) applied on the LEFT: world-frame yaw.
        """
        import math as _m
        zx, zy, zz, zw = 0.0, 0.0, _m.sin(theta / 2.0), _m.cos(theta / 2.0)
        x, y, z, w = q
        return (
            zw * x + zx * w + zy * z - zz * y,
            zw * y - zx * z + zy * w + zz * x,
            zw * z + zx * y - zy * x + zz * w,
            zw * w - zx * x - zy * y - zz * z,
        )

    def _run_standing_side_pick(self, target, obj_class, bx, by, bz):
        """Human-like SIDE grasp for standing objects (professor's request).

        The hand approaches HORIZONTALLY (with the template's slight downward
        tilt) at the object's body-middle height, like a person grabbing a
        bottle. The whole grasp (orientation + approach direction) is rotated
        by the object's yaw angle, so joint1 turns automatically for side
        objects via IK.
        """
        import math as _m
        # Yaw toward the object (base frame). Template was captured at yaw=0.
        theta = _m.atan2(by, bx)
        ux, uy = _m.cos(theta), _m.sin(theta)   # approach direction unit vec
        quat = self._yaw_rotated_quat(theta, self.side_quat_template)

        # Single grasp height for all standing objects. The objects in the
        # scene are of similar height, so one mid-body height serves all; the
        h = self.standing_side_grasp_height
        link6_z = bz + h + self.side_z_offset
        print(f'    [side] h={h:.3f}')

        # link6 positions: grasp point (behind object) and pre-approach point
        gx = bx - self.side_back_offset * ux
        gy = by - self.side_back_offset * uy
        px = bx - (self.side_back_offset + self.side_approach_extra) * ux
        py = by - (self.side_back_offset + self.side_approach_extra) * uy

        print(f'    [side] theta={_m.degrees(theta):.1f}deg  h={h:.3f}')
        print(f'    [side] pre=({px:.3f},{py:.3f},{link6_z:.3f}) '
              f'grasp=({gx:.3f},{gy:.3f},{link6_z:.3f})')

        # --- Step 1: open gripper ---
        self._step('1/8 OPEN gripper')
        self.set_gripper(True)
        time.sleep(1.0)

        # --- Step 2: pre-approach BESIDE the object at body height ---
        self._step('2/8 PRE-APPROACH (beside object, hand horizontal)')
        if not self._goto_pose(px, py, link6_z, ori_tol=0.35, quat=quat):
            print('  [X] Pre-approach failed. Aborting sequence.'); return

        # --- Step 3: move straight IN to wrap the body ---
        self._step('3/8 MOVE IN (wrap the body)')
        if not self._goto_pose(gx, gy, link6_z, ori_tol=0.35, quat=quat):
            print('  [X] Move-in failed. Aborting sequence.'); return


        self._step('4/8 CLOSE gripper (grab)')
        self.set_gripper(False)
        time.sleep(1.0)

        # --- Step 5: raise to neutral centered pose ---
        self._step('5/8 RAISE to neutral pose')
        if not self._goto_joints(self.raise_joints):
            print('  [X] Raise failed. Aborting sequence.'); return

        # Verify the object actually lifted (attachment succeeded). If it is
        # still on the table, the attach failed: skip the bin and go home.
        time.sleep(0.5)   # let perception update after the lift
        if self._object_still_on_table(bx, by):
            print('  [X] Object not attached (still on table). '
                  'Skipping bin, returning home.')
            self._goto_joints(self.home_joints)
            return

        # --- Step 6: ABOVE the correct bin (REUSED) ---
        bin_key = 'bottle' if obj_class == 'bottle' else 'can'
        poses = self.bin_poses[bin_key]
        self._step(f'6/8 ABOVE {bin_key} bin')
        if not self._goto_joints(poses['above']):
            print('  [X] Above-bin failed. Aborting sequence.'); return

        # --- Step 7: DROP + RELEASE (REUSED) ---
        self._step(f'7/8 DROP into {bin_key} bin')
        if not self._goto_joints(poses['drop']):
            print('  [X] Drop-pose failed. Aborting sequence.'); return
        self._step('RELEASE (open gripper)')
        time.sleep(1.0)
        self.set_gripper(True)
        time.sleep(1.0)

        # --- Step 7b: RISE out of the bin before swinging to home (bug fix) ---
        #     After release the fingers are deep in the bin. Go back UP to the
        #     bin 'above' pose, then to the centred neutral pose, so the fingers
        #     clear the bin edge instead of hitting it on the way home.
        self._step('7b/8 RISE out of bin')
        if not self._goto_joints(poses['above']):
            print('  [X] Rise-out-of-bin failed. Aborting sequence.'); return
        if not self._goto_joints(self.raise_joints):
            print('  [X] Raise-to-neutral failed. Aborting sequence.'); return

        # --- Step 8: HOME (REUSED) ---
        self._step('8/8 HOME')
        if not self._goto_joints(self.home_joints):
            print('  [X] Return-home failed.'); return

        self._seq_ok = True
        print('\n  >>> SIDE-GRASP PICK-AND-PLACE SEQUENCE COMPLETE <<<')
        print('=' * 60 + '\n')

    def _run_lying_pick(self, target, obj_class, bx, by, bz):
        """Lying-object pick-place-sort.

        IDENTICAL approach to standing (IK, gripper pointing DOWN) until the
        hand is right above/on the object. ONLY THEN do we twist joint6 so the
        fingers close ACROSS the round body. All other joints are frozen
        during the twist (no IK re-solve -> no arm twist).

        After the grab: raise -> bin -> drop -> release -> home (reused).
        """
        # Halved tilt compensation for the lying grasp: the offsets were
        # measured for the top-down orientation, but this sequence finishes
        # with the wrist twisted and descending in joint space, so the full
        # correction displaces the grasp off the object's centre.
        gx = bx - self.tilt_offset_x * 0.5
        gy = by - self.tilt_offset_y * 0.5
        print(f'    [lying] object=({bx:.3f},{by:.3f}) target=({gx:.3f},{gy:.3f})')

        # Heights for link6 (lying object center bz is low, ~1 radius up).
        pre_z = bz + self.lying_pregrasp_height + self.gripper_length
        twist_z = bz + self.lying_twist_clearance + self.gripper_length

        print(f'    [lying] gx={gx:.3f} gy={gy:.3f} '
              f'pre_z={pre_z:.3f} twist_z={twist_z:.3f} '
              f'wrist={self.lying_wrist_angle:.4f}')

        # --- Step 1: open gripper ---
        self._step('1/9 OPEN gripper')
        self.set_gripper(True)
        time.sleep(1.0)

        # --- Step 2: pre-grasp (hover above object, pointing DOWN) ---
        #     SAME as standing: IK + down orientation. Hand points at object.
        self._step('2/9 PRE-GRASP (hover above object, pointing down)')
        if not self._goto_pose(gx, gy, pre_z):
            print('  [X] Pre-grasp failed. Aborting sequence.'); return

        # --- Step 3: descend to SAFE TWIST height (IK, pointing DOWN) ---
        #     Stop ABOVE the body (lying_twist_clearance) so that when the
        #     fingers turn they do NOT sweep into a fat body (big bottle).
        self._step('3/9 DESCEND to twist height (pointing down)')
        if not self._goto_pose(gx, gy, twist_z):
            print('  [X] Descend-to-twist failed. Aborting sequence.'); return

        # --- Step 4: TWIST only joint6, FREEZE every other joint ---
        #     Fingers turn across the body while still clear above it.
        self._step('4/9 TWIST wrist (joint6 only)')
        cur = self._read_arm_joints()
        if cur is None:
            print('  [X] Could not read joint states. Aborting.'); return
        twisted = list(cur)              # keep joints 1..5 EXACTLY as IK set
        twisted[5] = self.lying_wrist_angle   # override ONLY joint6
        print(f'    reached joints: {[round(c,3) for c in cur]}')
        print(f'    wrist-twisted : {[round(c,3) for c in twisted]}')
        if not self._goto_joints(twisted):
            print('  [X] Wrist twist failed. Aborting sequence.'); return

        # --- Step 5: JOINT-SPACE DESCEND (no IK -> no 99999, twist held) ---
        #     Lower the hand by bending joints 2 & 3 a little, keeping joint6
        #     at the twist angle. Pure joint motion: the wrist stays turned
        #     and there is no IK fight. The fingers come down already across
        #     the body, so the fat bottle is not knocked away.
        self._step('5/9 JOINT-SPACE DESCEND onto object')
        descend = list(twisted)
        for _ in range(max(1, int(self.lying_descend_steps))):
            descend[1] += self.lying_descend_j2
            descend[2] += self.lying_descend_j3
        descend[5] = self.lying_wrist_angle
        print(f'    descend joints: {[round(c,3) for c in descend]}')
        if not self._goto_joints(descend):
            print('  [X] Joint-space descend failed. Aborting sequence.'); return

        # --- Step 6: close gripper (grab) ---
        self._step('6/10 CLOSE gripper (grab)')
        self.set_gripper(False)
        time.sleep(1.0)

        # --- Step 7: raise to neutral centered pose (avoids twist) ---
        self._step('7/10 RAISE to neutral pose')
        if not self._goto_joints(self.raise_joints):
            print('  [X] Raise failed. Aborting sequence.'); return

        # Verify the object actually lifted (attachment succeeded). If it is
        # still on the table, the attach failed: skip the bin and go home.
        time.sleep(0.5)   # let perception update after the lift
        if self._object_still_on_table(bx, by):
            print('  [X] Object not attached (still on table). '
                  'Skipping bin, returning home.')
            self._goto_joints(self.home_joints)
            return

        # --- Step 8: ABOVE the correct bin (REUSED from standing) ---
        bin_key = 'bottle' if obj_class == 'bottle' else 'can'
        poses = self.bin_poses[bin_key]
        self._step(f'8/10 ABOVE {bin_key} bin')
        if not self._goto_joints(poses['above']):
            print('  [X] Above-bin failed. Aborting sequence.'); return

        # --- Step 9: DROP + RELEASE (REUSED from standing) ---
        self._step(f'9/10 DROP into {bin_key} bin')
        if not self._goto_joints(poses['drop']):
            print('  [X] Drop-pose failed. Aborting sequence.'); return
        self._step('RELEASE (open gripper)')
        time.sleep(1.0)
        self.set_gripper(True)
        time.sleep(1.0)

        # --- Step 9b: RISE out of the bin before home (bug fix) ---
        self._step('9b/10 RISE out of bin')
        if not self._goto_joints(poses['above']):
            print('  [X] Rise-out-of-bin failed. Aborting sequence.'); return
        if not self._goto_joints(self.raise_joints):
            print('  [X] Raise-to-neutral failed. Aborting sequence.'); return

        # --- Step 10: HOME (REUSED from standing) ---
        self._step('10/10 HOME')
        if not self._goto_joints(self.home_joints):
            print('  [X] Return-home failed.'); return

        self._seq_ok = True
        print('\n  >>> LYING PICK-AND-PLACE SEQUENCE COMPLETE <<<')
        log_event('motion', 'sequence_complete_ok',
                  trial_id=target.label, target=target.label, pose='lying')
        print('=' * 60 + '\n')

    def _step(self, text):
        print(f'\n  --- STEP {text} ---')

    def world_to_base(self, wx, wy, wz):
        # base_link is rotated 90 deg about Z relative to Isaac world:
        # base_x = world_dY, base_y = -world_dX, base_z = world_dZ
        bx, by, bz = self.robot_base_world
        dx = wx - bx
        dy = wy - by
        dz = wz - bz
        return (dy, -dx, dz)

    # =================================================================
    def _find_target(self, label):
        if self.latest_vision is None or not label:
            return None
        want = label.strip().lower()
        for obj in self.latest_vision.objects:
            if obj.label.strip().lower() == want:
                return obj
        return None


    def _object_still_on_table(self, bx, by):
        # NOTE: returns True if ANY detected object lies within 0.10 m of the
        # grasp point (bx, by) — it does not verify that the TARGET itself
        # lifted. With well-separated objects this is a reliable "attach
        # failed" check; the 0.10 m radius is an empirical compromise between
        # missing a failed lift and false-positiving on a nearby object.
        if self.latest_vision is None:
            return False
        for obj in self.latest_vision.objects:
            ox, oy, _ = self.world_to_base(
                obj.world_x, obj.world_y, obj.world_z)
            dist = ((ox - bx) ** 2 + (oy - by) ** 2) ** 0.5
            if dist < 0.10:
                return True
        return False

    # ==================================================================
    def set_gripper(self, opened):
        """Open/close gripper by publishing joint7/joint8 to /joint_command."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint7', 'joint8']
        msg.position = self.grip_open if opened else self.grip_close
        for _ in range(5):
            self.gripper_pub.publish(msg)
            time.sleep(0.05)
        self.get_logger().info(f'Gripper -> {"OPEN" if opened else "CLOSE"}')

    # ==================================================================
    def _goto_pose(self, x, y, z, orient=True, ori_tol=0.2, quat=None):
        q = quat if quat is not None else self.down_quat
        goal = self._build_pose_goal(x, y, z, q, orient, ori_tol)
        return self._send_and_wait(goal, f'pose ({x:.3f},{y:.3f},{z:.3f})')

    def _goto_joints(self, joints):
        """Send a joint goal and BLOCK until done."""
        goal = self._build_joint_goal(joints)
        return self._send_and_wait(goal, f'joints {joints}')

    # ==================================================================
    def _build_pose_goal(self, x, y, z, quat, orient=True, ori_tol=0.2):
        pose = PoseStamped()
        pose.header.frame_id = self.planning_frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.planning_frame
        pos_c.link_name = self.eef_link
        pos_c.weight = 1.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.01]
        bv = BoundingVolume()
        bv.primitives.append(region)
        bv.primitive_poses.append(pose.pose)
        pos_c.constraint_region = bv

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.planning_frame
        ori_c.link_name = self.eef_link
        ori_c.orientation = pose.pose.orientation
        ori_c.absolute_x_axis_tolerance = ori_tol
        ori_c.absolute_y_axis_tolerance = ori_tol
        ori_c.absolute_z_axis_tolerance = ori_tol
        ori_c.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        if orient:
            constraints.orientation_constraints.append(ori_c)

        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.goal_constraints.append(constraints)
        goal.request.num_planning_attempts = 20
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only = False
        return goal

    def _build_joint_goal(self, joints):
        names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        constraints = Constraints()
        for name, pos in zip(names, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.goal_constraints.append(constraints)
        goal.request.num_planning_attempts = 20
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only = False
        return goal

    # ==================================================================
    def _send_and_wait(self, goal, desc, timeout=30.0):
        """Send a MoveGroup goal and block until the result arrives.
        Returns True on success (or control-handshake -6), False otherwise.
        """
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/move_action not available!')
            return False

        self._result_code = None
        self._result_event.clear()

        self.get_logger().info(f'Sending goal: {desc}')
        fut = self.move_client.send_goal_async(goal)
        fut.add_done_callback(self._goal_response_cb)

        # Block this sequence thread until the result callback fires.
        if not self._result_event.wait(timeout=timeout):
            self.get_logger().warn(f'Timeout waiting for: {desc}')
            return False

        code = self._result_code
        if code == 1:
            self.get_logger().info(f'  OK ({desc})')
            return True
        elif code == -6:
            # CONTROL_FAILED: arm moved, handshake imperfect. Treat as done.
            self.get_logger().info(f'  OK [-6 handshake] ({desc})')
            return True
        else:
            self.get_logger().warn(f'  FAILED code={code} ({desc})')
            return False

    # ------------------------------------------------------------------
    def _goal_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('MoveIt rejected the goal.')
            self._result_code = -999
            self._result_event.set()
            return
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._result_code = future.result().result.error_code.val
        self._result_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = MotionAgentNode()
    # MultiThreadedExecutor so the sequence thread can block on results
    # while subscriptions / action callbacks keep being serviced.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()