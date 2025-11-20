// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:msg/QueryPoints.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_H_
#define TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_H_

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
// Member 'points'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/QueryPoints in the package tracker_interfaces.
typedef struct tracker_interfaces__msg__QueryPoints
{
  std_msgs__msg__Header header;
  /// (N, 2) coordinates in [0,1] normalized
  geometry_msgs__msg__Point__Sequence points;
} tracker_interfaces__msg__QueryPoints;

// Struct for a sequence of tracker_interfaces__msg__QueryPoints.
typedef struct tracker_interfaces__msg__QueryPoints__Sequence
{
  tracker_interfaces__msg__QueryPoints * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__msg__QueryPoints__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_H_
