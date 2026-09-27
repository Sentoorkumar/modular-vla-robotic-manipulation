# VisionResult.msg
# =================
# Output of the perception node. Contains:
#   - A list of all detected objects with full info (DetectedObject[])
#   - Scene summary text (for logging / speech feedback)
#   - Disambiguation flags (helps LLM decide if clarification needed)
#   - Legacy fields kept for backward compatibility
#
# Published on: /perception/result

# --- Header for timestamp / frame_id ---
std_msgs/Header header

# ==================================================================
# NEW: Rich structured detections (use this in new code)
# ==================================================================
DetectedObject[] objects       # Array of all detected objects
int32 num_objects              # Total count of detected objects
int32 num_reachable            # Count of objects within robot's reach

# ==================================================================
# NEW: Disambiguation info for the LLM Coordinator
# ==================================================================
bool is_ambiguous              # True if >1 object of same class detected
string[] ambiguous_classes     # List of classes with multiple instances
                               #   e.g. ["bottle"] if 3 bottles, 1 can

# ==================================================================
# Scene summary (human-readable)
# ==================================================================
string scene_description       # e.g. "I see 2 bottles and 1 can"

# ==================================================================
# LEGACY FIELDS (kept for backward compatibility with existing nodes)
# ==================================================================
# These are populated from 'objects' for any nodes that still read
# the old interface. New code should read 'objects' instead.
string[] detected_objects      # list of class names (duplicates allowed)
float32[] confidence_scores    # parallel to detected_objects
bool graspable                 # true if at least one object is reachable
string graspability_reason     # explanation text
string intent_type             # filled by LLM (not perception) - kept for compatibility
string target_object           # filled by LLM (not perception) - kept for compatibility
