// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tracker_interfaces:srv/SetPredictionPrompt.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__STRUCT_H_
#define TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__STRUCT_H_

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

/// Struct defined in srv/SetPredictionPrompt in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__SetPredictionPrompt_Request
{
  rosidl_runtime_c__String prompt;
} tracker_interfaces__srv__SetPredictionPrompt_Request;

// Struct for a sequence of tracker_interfaces__srv__SetPredictionPrompt_Request.
typedef struct tracker_interfaces__srv__SetPredictionPrompt_Request__Sequence
{
  tracker_interfaces__srv__SetPredictionPrompt_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__SetPredictionPrompt_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetPredictionPrompt in the package tracker_interfaces.
typedef struct tracker_interfaces__srv__SetPredictionPrompt_Response
{
  bool success;
  rosidl_runtime_c__String message;
} tracker_interfaces__srv__SetPredictionPrompt_Response;

// Struct for a sequence of tracker_interfaces__srv__SetPredictionPrompt_Response.
typedef struct tracker_interfaces__srv__SetPredictionPrompt_Response__Sequence
{
  tracker_interfaces__srv__SetPredictionPrompt_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tracker_interfaces__srv__SetPredictionPrompt_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__STRUCT_H_
