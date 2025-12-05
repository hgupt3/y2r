std_msgs/Header header
float32[] trajectory_x                 # Flattened (N*T,) x-coordinates in camera pixels
float32[] trajectory_y                 # Flattened (N*T,) y-coordinates in camera pixels
float32[] trajectory_z                 # Flattened (N*T,) z-coordinates (depth, 3D only)
int32 num_points                       # N
int32 num_timesteps                    # T
