// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_H_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'left_trajectory_u'
// Member 'left_trajectory_v'
// Member 'left_trajectory_d'
// Member 'right_trajectory_u'
// Member 'right_trajectory_v'
// Member 'right_trajectory_d'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/PredictedHandPoses in the package tracker_interfaces.
typedef struct tracker_interfaces__msg__PredictedHandPoses
{
  std_msgs__msg__Header header;
  int32_t num_timesteps;
  /// Left hand
  bool left_valid;
  /// (T,) camera pixel u
  rosidl_runtime_c__float__Sequence left_trajectory_u;
  /// (T,) camera pixel v
  rosidl_runtime_c__float__Sequence left_trajectory_v;
  /// (T,) depth in meters
  rosidl_runtime_c__float__Sequence left_trajectory_d;
  /// Final 3x3 rotation (flattened)
  float left_final_rotation[9];
  /// Right hand
  bool right_valid;
  rosidl_runtime_c__float__Sequence right_trajectory_u;
  rosidl_runtime_c__float__Sequence right_trajectory_v;
  rosidl_runtime_c__float__Sequence right_trajectory_d;
  float right_final_rotation[9];
} tracker_interfaces__msg__PredictedHandPoses;

// Struct for a sequence of tracker_interfaces__msg__PredictedHandPoses.
typedef struct tracker_interfaces__msg__PredictedHandPoses__Sequence
{
  tracker_interfaces__msg__PredictedHandPoses * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__msg__PredictedHandPoses__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_H_
