// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:srv/ResetTracking.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_H_
#define TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/ResetTracking in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__ResetTracking_Request
{
  uint8_t structure_needs_at_least_one_member;
} tracker_interfaces__srv__ResetTracking_Request;

// Struct for a sequence of tracker_interfaces__srv__ResetTracking_Request.
typedef struct tracker_interfaces__srv__ResetTracking_Request__Sequence
{
  tracker_interfaces__srv__ResetTracking_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__ResetTracking_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/ResetTracking in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__ResetTracking_Response
{
  bool success;
  rosidl_runtime_c__String message;
} tracker_interfaces__srv__ResetTracking_Response;

// Struct for a sequence of tracker_interfaces__srv__ResetTracking_Response.
typedef struct tracker_interfaces__srv__ResetTracking_Response__Sequence
{
  tracker_interfaces__srv__ResetTracking_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__ResetTracking_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_H_
