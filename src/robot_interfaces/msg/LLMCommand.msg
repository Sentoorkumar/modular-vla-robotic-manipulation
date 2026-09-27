# Command from LLM Coordinator to Agents
string intent
string target_object
string reasoning

# --- Bulk clear/sort filters (used only when intent == "clear_all") ---
# Empty string means "no filter" (match any). Ignored for normal picks.
string filter_class       # "bottle" | "can" | ""  (any class)
string filter_pose        # "standing" | "lying" | ""  (any pose)