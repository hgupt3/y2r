std_msgs/Header header

int32 num_timesteps

# Left hand
bool left_valid
float32[] left_trajectory_u    # (T,) camera pixel u
float32[] left_trajectory_v    # (T,) camera pixel v
float32[] left_trajectory_d    # (T,) depth in meters
float32[9] left_final_rotation # Final 3x3 rotation (flattened)

# Right hand
bool right_valid
float32[] right_trajectory_u
float32[] right_trajectory_v
float32[] right_trajectory_d
float32[9] right_final_rotation

