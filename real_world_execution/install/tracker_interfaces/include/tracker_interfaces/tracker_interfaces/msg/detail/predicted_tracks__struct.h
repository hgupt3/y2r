// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_H_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_H_

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
// Member 'query_points'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'trajectory_x'
// Member 'trajectory_y'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/PredictedTracks in the package tracker_interfaces.
typedef struct tracker_interfaces__msg__PredictedTracks
{
  std_msgs__msg__Header header;
  /// (N, 2) starting positions
  geometry_msgs__msg__Point__Sequence query_points;
  /// Flattened (N*T,) x-coordinates
  rosidl_runtime_c__float__Sequence trajectory_x;
  /// Flattened (N*T,) y-coordinates
  rosidl_runtime_c__float__Sequence trajectory_y;
  /// N
  int32_t num_points;
  /// T
  int32_t num_timesteps;
} tracker_interfaces__msg__PredictedTracks;

// Struct for a sequence of tracker_interfaces__msg__PredictedTracks.
typedef struct tracker_interfaces__msg__PredictedTracks__Sequence
{
  tracker_interfaces__msg__PredictedTracks * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__msg__PredictedTracks__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_H_
