# =============================================================================
# grasp_with_judge.py  —  Isaac Sim Script Editor
# Attachment-based grasp + success judge for the Pfand sorting system.
#
# GRASP:   on gripper close, the nearest object within its attach radius is
#          joined to link6 by a temporary fixed joint, held at a fixed offset
#          on the link6 side (so object prim scale cannot distort it).
# RELEASE: when the gripper is opened AND the arm is near a bin.
# SAFETY:  if an object is held longer than MAX_HOLD_TIME, it is dropped.
#
# JUDGE:   snapshot_start("standing"|"lying")  ->  (send command)  ->  judge()
# =============================================================================

import omni.usd
import time as _time
import os as _os
import csv as _csv
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
from omni.physx import get_physx_interface

# ---------------------------------------------------------------- ROBOT PATHS
GRIPPER_LINK = "/Environment/piper_ros2_working/piper_v1/piper_camera/link6"
ATTACH_LINK  = "/Environment/piper_ros2_working/piper_v1/piper_camera/link6"
JOINT7_PATH  = "/Environment/piper_ros2_working/piper_v1/piper_camera/joints/joint7"

OBJECT_PATHS = [
    "/World/_07_tuna_fish_can",
    "/World/NaturalBostonRoundBottle_A01_PR_NVD_01",
    "/World/WhitePackerBottle_A01_PR_NVD_01",
    "/World/_05_tomato_soup_can",
    "/World/_05_tomato_soup_can_01",
]

# ---------------------------------------------------------------- GRASP CONFIG
FIXED_JOINT_PATH = "/World/grasp_fixed_joint"

CLOSE_THRESHOLD = 0.025   # joint7 below this  -> fingers closed  -> try attach
OPEN_THRESHOLD  = 0.040   # joint7 above this  -> fingers open     -> release

# Default attach radius (link6 -> object origin). Kept tight so that objects
# standing close together are not attached by mistake.
ATTACH_RADIUS = 0.20

# Per-object override. The tall bottle's prim origin sits at its base, so link6
# is farther from that origin when gripping its mid-body; it needs a larger
# radius, applied ONLY to this object rather than widening the global default.
ATTACH_RADIUS_PER_OBJECT = {
    "/World/NaturalBostonRoundBottle_A01_PR_NVD_01": 0.28,
}

# Offset (in link6's frame) at which a grasped object is held, so it sits
# between the fingers regardless of grasp direction. Applied on the LINK6 side
# of the joint: link6 has unit scale, whereas some objects were resized with a
# non-uniform prim scale that would otherwise distort this offset.
HOLD_OFFSET = Gf.Vec3f(0.0, 0.0, 0.12)

# Release: the gripper must be open AND near a bin.
BIN_POSITIONS = [
    Gf.Vec3d(0.45, -0.65, 2.50),    # right bin (cans)
    Gf.Vec3d(-0.45, -0.65, 2.50),   # left bin (bottles)
]
BIN_RELEASE_DIST = 0.30

# After releasing, ignore re-attachment for this long so the just-dropped object
# is not immediately grabbed again while it is still falling into the bin.
RELEASE_COOLDOWN = 10.0

# Safety: if an object has been held far longer than a normal pick takes,
# something went wrong -> drop it rather than leave it stuck to the arm.
# This is time-based (not position-based) so it cannot fire mid-sequence.
MAX_HOLD_TIME = 30.0

# ---------------------------------------------------------------- JUDGE CONFIG
TABLE_TOP_Z = 2.427

_CAN_BIN    = Gf.Vec3d(-0.45, -0.65, 2.50)   # cans belong here
_BOTTLE_BIN = Gf.Vec3d(0.45, -0.65, 2.50)    # bottles belong here

OBJECT_CLASS = {
    "/World/_07_tuna_fish_can":                       "can",
    "/World/_05_tomato_soup_can":                     "can",
    "/World/_05_tomato_soup_can_01":                  "can",
    "/World/NaturalBostonRoundBottle_A01_PR_NVD_01":  "bottle",
    "/World/WhitePackerBottle_A01_PR_NVD_01":         "bottle",
}

OBJECT_TYPES = {
    "/World/_07_tuna_fish_can":                      "tuna_can",
    "/World/NaturalBostonRoundBottle_A01_PR_NVD_01": "big_bottle",
    "/World/WhitePackerBottle_A01_PR_NVD_01":        "packer_bottle",
    "/World/_05_tomato_soup_can":                    "soup_can",
    "/World/_05_tomato_soup_can_01":                 "soup_can_2",
}

BIN_XY_RADIUS   = 0.30    # within this of a bin centre => delivered
LIFT_THRESHOLD  = 0.06    # rose this far above the table => lifted
FLOOR_THRESHOLD = 0.25    # fell this far below the table => on floor
MOVED_THRESHOLD = 0.08    # moved less than this => still on table

_LOG_FILE = _os.environ.get(
    'THESIS_METRICS_FILE',
    _os.path.expanduser('~/thesis_metrics/run_log.csv')
)
_LOG_HEADER = ['wall_time_iso', 'unix_time', 'node', 'event',
               'trial_id', 'target', 'pose', 'extra']

# ---------------------------------------------------------------- STATE
stage = omni.usd.get_context().get_stage()
_state = {"attached": False, "obj": None,
          "release_time": 0.0, "attach_time": 0.0}
_judge = {"start": {}, "maxz": {}, "trial": 0, "pose": ""}


# ================================================================ HELPERS
def _world_pos(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    return Gf.Vec3d(t[0], t[1], t[2])


def _read_joint7():
    prim = stage.GetPrimAtPath(JOINT7_PATH)
    if not prim or not prim.IsValid():
        return None
    a = prim.GetAttribute("state:linear:physics:position")
    return a.Get() if (a and a.IsValid()) else None


def _nearest_object():
    gprim = stage.GetPrimAtPath(GRIPPER_LINK)
    if not gprim or not gprim.IsValid():
        return None, 1e9
    m = UsdGeom.Xformable(gprim).ComputeLocalToWorldTransform(0)
    # Point between the fingers, not the wrist: transform HOLD_OFFSET from
    # link6's frame into world. Measuring from here keeps the attach gate tied
    # to where the fingers actually are, rather than to the wrist pose, which
    # drifts as the arm extends toward the table edges.
    gpos = m.Transform(Gf.Vec3d(HOLD_OFFSET[0], HOLD_OFFSET[1], HOLD_OFFSET[2]))
    best, best_d = None, 1e9
    for op in OBJECT_PATHS:
        opos = _world_pos(op)
        if opos is None:
            continue
        d = (opos - gpos).GetLength()
        if d < best_d:
            best, best_d = op, d
    return best, best_d


def _near_bin():
    gpos = _world_pos(GRIPPER_LINK)
    if gpos is None:
        return False
    for b in BIN_POSITIONS:
        if (gpos - b).GetLength() < BIN_RELEASE_DIST:
            return True
    return False


# ================================================================ GRASP
def _attach():
    if _state["attached"]:
        return
    gpos = _world_pos(GRIPPER_LINK)
    if gpos is None:
        return
    # Check EVERY object against its OWN radius, then take the closest match.
    # (Checking only the single nearest object can miss the tall bottle, whose
    # origin sits at its base and so is farther away than a neighbouring one.)
    obj, dist = None, 1e9
    for op in OBJECT_PATHS:
        opos = _world_pos(op)
        if opos is None:
            continue
        d = (opos - gpos).GetLength()
        r = ATTACH_RADIUS_PER_OBJECT.get(op, ATTACH_RADIUS)
        if d <= r and d < dist:
            obj, dist = op, d
    if obj is None:
        print("[grasp] no object within its attach radius")
        return

    if stage.GetPrimAtPath(FIXED_JOINT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(FIXED_JOINT_PATH))

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(FIXED_JOINT_PATH))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(ATTACH_LINK)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(obj)])
    # Offset on the link6 side; object attached at its own origin.
    joint.CreateLocalPos0Attr().Set(HOLD_OFFSET)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateJointEnabledAttr().Set(True)

    _state["attached"] = True
    _state["obj"] = obj
    _state["attach_time"] = _time.time()
    print(f"[grasp] ATTACHED {obj} (dist {dist:.3f}m)")


def _detach():
    if not _state["attached"]:
        return
    jp = stage.GetPrimAtPath(FIXED_JOINT_PATH)
    if jp and jp.IsValid():
        joint = UsdPhysics.Joint(jp)
        if joint:
            en = joint.GetJointEnabledAttr()
            if not en or not en.IsValid():
                en = joint.CreateJointEnabledAttr()
            en.Set(False)
            joint.GetBody0Rel().SetTargets([])
            joint.GetBody1Rel().SetTargets([])
        stage.RemovePrim(Sdf.Path(FIXED_JOINT_PATH))
    obj = _state.get("obj")
    if obj:
        op = stage.GetPrimAtPath(obj)
        if op and op.IsValid():
            rb = UsdPhysics.RigidBodyAPI(op)
            if rb:
                v = rb.GetVelocityAttr()
                if v:
                    v.Set(Gf.Vec3f(0, 0, 0))
    _state["attached"] = False
    _state["obj"] = None
    _state["release_time"] = _time.time()
    print("[grasp] DETACHED")


def reset_grasp():
    """Manual recovery: force-remove any stuck joint and clear the state."""
    if stage.GetPrimAtPath(FIXED_JOINT_PATH).IsValid():
        try:
            j = UsdPhysics.Joint(stage.GetPrimAtPath(FIXED_JOINT_PATH))
            if j:
                en = j.GetJointEnabledAttr()
                if not en or not en.IsValid():
                    en = j.CreateJointEnabledAttr()
                en.Set(False)
                j.GetBody0Rel().SetTargets([])
                j.GetBody1Rel().SetTargets([])
            stage.RemovePrim(Sdf.Path(FIXED_JOINT_PATH))
        except Exception as e:
            print("[grasp] reset error:", e)
    _state["attached"] = False
    _state["obj"] = None
    _state["release_time"] = 0.0
    _state["attach_time"] = 0.0
    print("[grasp] reset: joint removed, state cleared")


# ================================================================ JUDGE
def _ensure_csv():
    try:
        _os.makedirs(_os.path.dirname(_LOG_FILE), exist_ok=True)
        if not _os.path.exists(_LOG_FILE) or _os.path.getsize(_LOG_FILE) == 0:
            with open(_LOG_FILE, 'w', newline='') as f:
                _csv.writer(f).writerow(_LOG_HEADER)
    except Exception as e:
        print("[judge] csv init error:", e)


def _log_row(event, target, extra):
    try:
        _ensure_csv()
        row = [
            _time.strftime('%Y-%m-%dT%H:%M:%S'),
            f'{_time.time():.6f}',
            'isaac', event, str(_judge["trial"]),
            str(target), '', str(extra),
        ]
        with open(_LOG_FILE, 'a', newline='') as f:
            _csv.writer(f).writerow(row)
    except Exception as e:
        print("[judge] log error:", e)


def snapshot_start(pose=""):
    """Record current object positions as the 'before' for a new trial.
    Tag the trial with the pose being tested, e.g. snapshot_start("standing").
    """
    _judge["trial"] += 1
    _judge["pose"] = pose
    _judge["start"] = {}
    _judge["maxz"] = {}
    for p in OBJECT_PATHS:
        pos = _world_pos(p)
        if pos is not None:
            _judge["start"][p] = pos
            _judge["maxz"][p] = pos[2]
    tag = f' pose="{pose}"' if pose else ""
    print(f"[judge] snapshot for trial {_judge['trial']}{tag} "
          f"({len(_judge['start'])} objects)")


def _classify(path):
    start = _judge["start"].get(path)
    end = _world_pos(path)
    maxz = _judge["maxz"].get(path, TABLE_TOP_Z)
    if start is None or end is None:
        return "unknown", 0, start, end, maxz

    lifted = 1 if (maxz - TABLE_TOP_Z) >= LIFT_THRESHOLD else 0
    if end[2] < (TABLE_TOP_Z - FLOOR_THRESHOLD):
        return "on_floor", lifted, start, end, maxz

    def _xy(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    cls = OBJECT_CLASS.get(path, "")
    d_can = _xy(end, _CAN_BIN)
    d_bottle = _xy(end, _BOTTLE_BIN)
    moved = _xy(end, start)

    # Only count as delivered if the object was actually lifted and moved:
    # otherwise an object left standing near a bin is scored as sorted
    # without the arm having touched it.
    if (d_can <= BIN_XY_RADIUS or d_bottle <= BIN_XY_RADIUS) \
       and lifted and moved >= MOVED_THRESHOLD:
        landed = "can" if d_can <= d_bottle else "bottle"
        if cls and landed == cls:
            return "sorted_correct", lifted, start, end, maxz
        return "sorted_wrong_bin", lifted, start, end, maxz

    if lifted and moved >= MOVED_THRESHOLD:
        return "lifted_only", lifted, start, end, maxz
    if moved < MOVED_THRESHOLD:
        return "on_table", lifted, start, end, maxz
    return "moved_on_table", lifted, start, end, maxz


def judge():
    """Classify every object against its snapshot and log the changed ones."""
    if not _judge["start"]:
        print("[judge] no snapshot yet — call snapshot_start() first")
        return
    pose = _judge.get("pose", "")
    print(f'\n[judge] ===== TRIAL {_judge["trial"]} RESULT'
          f'{" pose=" + pose if pose else ""} =====')
    logged = 0
    for p in OBJECT_PATHS:
        outcome, lifted, start, end, maxz = _classify(p)
        s = f"({start[0]:.2f},{start[1]:.2f},{start[2]:.2f})" if start else "NA"
        e = f"({end[0]:.2f},{end[1]:.2f},{end[2]:.2f})" if end else "NA"
        obj_type = OBJECT_TYPES.get(p, p.split('/')[-1])
        print(f"  {p.split('/')[-1]:40s} -> {outcome:16s} lifted={lifted}")

        if outcome not in ("on_table", "unknown"):
            extra = (f"outcome={outcome};lifted={lifted};"
                     f"objtype={obj_type};pose={pose};"
                     f"start={s};end={e};maxz={maxz:.3f}")
            _log_row('trial_result', p, extra)
            logged += 1

    if logged == 0:
        _log_row('trial_result', 'NONE',
                 f"outcome=no_change;lifted=0;objtype=none;pose={pose}")
    print(f"[judge] logged {logged} changed object(s)")
    print("[judge] ===== END =====\n")


# ================================================================ PHYSICS STEP
def _on_physics_step(dt):
    # Track the maximum height each object reaches (for the lift test).
    if _judge["start"]:
        for _p in _judge["start"].keys():
            _pos = _world_pos(_p)
            if _pos is not None and _pos[2] > _judge["maxz"].get(_p, -1e9):
                _judge["maxz"][_p] = _pos[2]

    j7 = _read_joint7()
    if j7 is None:
        return

    # RELEASE: near a bin -> detach, regardless of the finger reading.
    # The finger joint does not settle to a predictable value on wide objects,
    # so bin proximity alone is used as the release condition.
    if _state["attached"] and _near_bin():
        print(f"[grasp] release at bin (j7={j7:.3f})")
        _detach()
        return

    # SAFETY: held far longer than a normal pick -> drop it.
    if _state["attached"] and \
       (_time.time() - _state.get("attach_time", 0.0)) > MAX_HOLD_TIME:
        print("[grasp] safety drop (held too long)")
        _detach()
        return
        
    if _state["attached"]:
       _gp = _world_pos(GRIPPER_LINK)
       _db = min((_gp - b).GetLength() for b in BIN_POSITIONS) if _gp else -1
       print(f"[dbg] j7={j7:.3f} (open>{OPEN_THRESHOLD}) bin_dist={_db:.3f} (need<{BIN_RELEASE_DIST})") 

    # GRAB: fingers closed, nothing held, not at a bin, cooldown elapsed.
    if j7 < CLOSE_THRESHOLD and not _state["attached"]:
        if _near_bin():
            return
        if _state["release_time"] > 0.0 and \
           (_time.time() - _state["release_time"]) < RELEASE_COOLDOWN:
            return
        _attach()


# ================================================================ STARTUP
try:
    _grasp_sub.unsubscribe()
except Exception:
    pass

if stage.GetPrimAtPath(FIXED_JOINT_PATH).IsValid():
    try:
        j = UsdPhysics.Joint(stage.GetPrimAtPath(FIXED_JOINT_PATH))
        if j:
            j.GetBody0Rel().SetTargets([])
            j.GetBody1Rel().SetTargets([])
        stage.RemovePrim(Sdf.Path(FIXED_JOINT_PATH))
        print("[grasp] cleaned leftover joint on startup")
    except Exception:
        pass
_state["attached"] = False
_state["obj"] = None

_physx = get_physx_interface()
_grasp_sub = _physx.subscribe_physics_step_events(_on_physics_step)

print("[grasp] running.  attach<%.3f  release>%.3f  cooldown=%.0fs  maxhold=%.0fs"
      % (CLOSE_THRESHOLD, OPEN_THRESHOLD, RELEASE_COOLDOWN, MAX_HOLD_TIME))
print("[grasp] joint7 =", _read_joint7())
_ensure_csv()
print("[judge] measurement active. Log file:", _LOG_FILE)
print("[judge] Per trial:  snapshot_start('standing')  ->  command  ->  judge()")
print("[grasp] If an object ever gets stuck attached, run:  reset_grasp()")
