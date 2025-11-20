std_msgs/Header header
geometry_msgs/Point[] query_points     # (N, 2) starting positions  
float32[] trajectory_x                 # Flattened (N*T,) x-coordinates
float32[] trajectory_y                 # Flattened (N*T,) y-coordinates
int32 num_points                       # N
int32 num_timesteps                    # T

