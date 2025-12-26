std_msgs/Header header

# Left hand
float32 left_u              # Wrist u in camera pixels
float32 left_v              # Wrist v in camera pixels  
float32 left_depth          # Wrist depth in meters
float32[9] left_rotation    # Flattened 3x3 rotation matrix (row-major)
bool left_valid             # True if left hand detected

# Right hand  
float32 right_u             # Wrist u in camera pixels
float32 right_v             # Wrist v in camera pixels
float32 right_depth         # Wrist depth in meters
float32[9] right_rotation   # Flattened 3x3 rotation matrix (row-major)
bool right_valid            # True if right hand detected

