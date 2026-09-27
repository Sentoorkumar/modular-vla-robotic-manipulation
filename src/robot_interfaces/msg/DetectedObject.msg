# DetectedObject.msg
# ==================
# Represents a single object detected by the perception system,
# enriched with 3D position and reachability info.
#
# Used as array element inside VisionResult.msg.

# --- Identity ---
string label              # Unique label, e.g. "bottle 1", "can 2"
string object_class       # YOLO class name, e.g. "bottle" or "can"
float32 confidence        # YOLO detection confidence (0.0 - 1.0)

# --- 2D pixel info (for debug / visualization) ---
float32 pixel_u           # grounding pixel X (horizontal)
float32 pixel_v           # grounding pixel Y (vertical)
float32 bbox_x1           # bounding box top-left X
float32 bbox_y1           # bounding box top-left Y
float32 bbox_x2           # bounding box bottom-right X
float32 bbox_y2           # bounding box bottom-right Y

# --- 3D world info ---
float32 world_x           # position X in world frame (meters)
float32 world_y           # position Y in world frame (meters)
float32 world_z           # position Z in world frame (meters, = table_top_z)

# --- Pose classification ---
string pose               # "standing" or "lying"

# --- Reachability (relative to robot base) ---
float32 distance_from_robot   # horizontal distance from robot base (meters)
bool reachable                # true if within robot's max_reach
string reach_status           # "reachable" / "too_far" / "unknown"

