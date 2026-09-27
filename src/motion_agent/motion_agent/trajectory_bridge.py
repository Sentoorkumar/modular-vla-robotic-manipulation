#!/usr/bin/env python3
"""
trajectory_bridge.py
====================
Method-A bridge: MoveIt2 -> Isaac Sim.

MoveIt plans a trajectory and sends it via a FollowJointTrajectory action.
This node ACTS as that controller: it receives the trajectory, then streams
each waypoint to /joint_command (sensor_msgs/JointState), which Isaac Sim
executes. Isaac publishes /joint_states back as the real robot state.

Modes (param 'auto_execute'):
  False (default) -> MANUAL: print trajectory, wait for ENTER before sending.
  True            -> AUTO:   send immediately (final-pipeline behaviour).

Gripper note: this bridge moves ARM joints (joint1..joint6) only. It holds the
gripper (joint7, joint8) at their last-known values so Isaac doesn't snap them.
Gripper open/close is commanded separately (Step 6).
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration


ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINTS = ['joint7', 'joint8']


class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__('trajectory_bridge')

        # ---- Parameter: manual vs auto ----
        self.declare_parameter('auto_execute', False)
        self.auto_execute = self.get_parameter('auto_execute').value
        mode = 'AUTO' if self.auto_execute else 'MANUAL (waits for ENTER)'
        self.get_logger().info(f'Trajectory bridge starting. Mode: {mode}')

        # ---- Track latest gripper position from Isaac (so we hold it) ----
        self.last_gripper = {'joint7': 0.0, 'joint8': 0.0}
        self.have_gripper = False

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10
        )

        # ---- Publisher to Isaac ----
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_command', 10
        )

        # ---- Action server that MoveIt talks to ----
        # Name must match moveit_controllers.yaml:
        #   controller 'piper_arm_controller' + action_ns 'follow_joint_trajectory'
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/piper_arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            'Ready. Advertising action: '
            '/piper_arm_controller/follow_joint_trajectory'
        )

    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState):
        """Remember the latest gripper positions from Isaac."""
        for name, pos in zip(msg.name, msg.position):
            if name in self.last_gripper:
                self.last_gripper[name] = pos
        self.have_gripper = True

    # ------------------------------------------------------------------
    def _goal_cb(self, goal_request):
        self.get_logger().info('Received trajectory goal from MoveIt.')
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info('Cancel requested.')
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    def _execute_cb(self, goal_handle):
        """Stream the planned trajectory to /joint_command."""
        traj = goal_handle.request.trajectory
        joint_names = list(traj.joint_names)
        points = traj.points
        n = len(points)

        self.get_logger().info(
            f'Trajectory: {n} waypoints over joints {joint_names}'
        )

        # ---- MANUAL mode: print summary, wait for ENTER ----
        if not self.auto_execute:
            self._print_trajectory_summary(joint_names, points)
            self.get_logger().info(
                '>>> MANUAL MODE: press ENTER in the terminal to execute, '
                'or Ctrl+C to abort. <<<'
            )
            # Block this action thread until user presses Enter.
            # (ReentrantCallbackGroup lets the rest of the node keep spinning.)
            input()
            self.get_logger().info('ENTER received. Executing.')

        # ---- Stream waypoints, pacing by time_from_start ----
        prev_t = 0.0
        for i, pt in enumerate(points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled mid-execution.')
                result = FollowJointTrajectory.Result()
                return result

            t = self._dur_to_sec(pt.time_from_start)
            dt = t - prev_t
            prev_t = t
            if dt > 0:
                time.sleep(dt)

            self._publish_waypoint(joint_names, pt.positions)

            # progress feedback
            fb = FollowJointTrajectory.Feedback()
            goal_handle.publish_feedback(fb)

        # Make sure the FINAL waypoint is definitely sent
        if points:
            self._publish_waypoint(joint_names, points[-1].positions)

        self.get_logger().info('Trajectory execution complete.')
        goal_handle.succeed()

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    # ------------------------------------------------------------------
    def _publish_waypoint(self, joint_names, positions):
        """Publish one waypoint as a JointState to /joint_command.

        Includes the arm joints from the trajectory, plus the gripper joints
        held at their last-known value so Isaac doesn't snap them shut/open.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        names = list(joint_names)
        pos = list(positions)

        # Append gripper hold values
        for gj in GRIPPER_JOINTS:
            if gj not in names:
                names.append(gj)
                pos.append(self.last_gripper.get(gj, 0.0))

        msg.name = names
        msg.position = pos
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _dur_to_sec(d: Duration) -> float:
        return float(d.sec) + float(d.nanosec) * 1e-9

    def _print_trajectory_summary(self, joint_names, points):
        print('\n' + '=' * 70)
        print(f'  TRAJECTORY PREVIEW  ({len(points)} waypoints)')
        print(f'  Joints: {joint_names}')
        print('-' * 70)
        # print first, middle, last waypoint for a quick sanity check
        idxs = sorted(set([0, len(points) // 2, len(points) - 1]))
        for i in idxs:
            pt = points[i]
            t = self._dur_to_sec(pt.time_from_start)
            vals = ', '.join(f'{p:+.3f}' for p in pt.positions)
            print(f'  wp[{i:3d}] t={t:5.2f}s  [{vals}]')
        print('=' * 70 + '\n')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryBridge()

    # MultiThreadedExecutor so input() blocking in the action callback
    # doesn't freeze the whole node (joint_state updates keep flowing).
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