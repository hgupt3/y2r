// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_H_
#define TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_H_

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

/// Struct defined in msg/HandPoses in the package tracker_interfaces.
typedef struct tracker_interfaces__msg__HandPoses
{
  std_msgs__msg__Header header;
  /// Left hand
  /// Wrist u in camera pixels
  float left_u;
  /// Wrist v in camera pixels
  float left_v;
  /// Wrist depth in meters
  float left_depth;
  /// Flattened 3x3 rotation matrix (row-major)
  float left_rotation[9];
  /// True if left hand detected
  bool left_valid;
  /// Right hand
  /// Wrist u in camera pixels
  float right_u;
  /// Wrist v in camera pixels
  float right_v;
  /// Wrist depth in meters
  float right_depth;
  /// Flattened 3x3 rotation matrix (row-major)
  float right_rotation[9];
  /// True if right hand detected
  bool right_valid;
} tracker_interfaces__msg__HandPoses;

// Struct for a sequence of tracker_interfaces__msg__HandPoses.
typedef struct tracker_interfaces__msg__HandPoses__Sequence
{
  tracker_interfaces__msg__HandPoses * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__msg__HandPoses__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_H_
