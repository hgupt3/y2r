// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:srv/SetPrompt.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__SET_PROMPT__STRUCT_H_
#define TRACKER_INTERFACES__SRV__DETAIL__SET_PROMPT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'prompt'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetPrompt in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__SetPrompt_Request
{
  rosidl_runtime_c__String prompt;
} tracker_interfaces__srv__SetPrompt_Request;

// Struct for a sequence of tracker_interfaces__srv__SetPrompt_Request.
typedef struct tracker_interfaces__srv__SetPrompt_Request__Sequence
{
  tracker_interfaces__srv__SetPrompt_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__SetPrompt_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetPrompt in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__SetPrompt_Response
{
  bool success;
  rosidl_runtime_c__String message;
} tracker_interfaces__srv__SetPrompt_Response;

// Struct for a sequence of tracker_interfaces__srv__SetPrompt_Response.
typedef struct tracker_interfaces__srv__SetPrompt_Response__Sequence
{
  tracker_interfaces__srv__SetPrompt_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__SetPrompt_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__SRV__DETAIL__SET_PROMPT__STRUCT_H_
