// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__FUNCTIONS_H_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "tracker_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "tracker_interfaces/msg/detail/predicted_tracks__struct.h"

/// Initialize msg/PredictedTracks message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * tracker_interfaces__msg__PredictedTracks
 * )) before or use
 * tracker_interfaces__msg__PredictedTracks__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__init(tracker_interfaces__msg__PredictedTracks * msg);

/// Finalize msg/PredictedTracks message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
void
tracker_interfaces__msg__PredictedTracks__fini(tracker_interfaces__msg__PredictedTracks * msg);

/// Create msg/PredictedTracks message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * tracker_interfaces__msg__PredictedTracks__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
tracker_interfaces__msg__PredictedTracks *
tracker_interfaces__msg__PredictedTracks__create();

/// Destroy msg/PredictedTracks message.
/**
 * It calls
 * tracker_interfaces__msg__PredictedTracks__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
void
tracker_interfaces__msg__PredictedTracks__destroy(tracker_interfaces__msg__PredictedTracks * msg);

/// Check for msg/PredictedTracks message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__are_equal(const tracker_interfaces__msg__PredictedTracks * lhs, const tracker_interfaces__msg__PredictedTracks * rhs);

/// Copy a msg/PredictedTracks message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__copy(
  const tracker_interfaces__msg__PredictedTracks * input,
  tracker_interfaces__msg__PredictedTracks * output);

/// Initialize array of msg/PredictedTracks messages.
/**
 * It allocates the memory for the number of elements and calls
 * tracker_interfaces__msg__PredictedTracks__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__Sequence__init(tracker_interfaces__msg__PredictedTracks__Sequence * array, size_t size);

/// Finalize array of msg/PredictedTracks messages.
/**
 * It calls
 * tracker_interfaces__msg__PredictedTracks__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
void
tracker_interfaces__msg__PredictedTracks__Sequence__fini(tracker_interfaces__msg__PredictedTracks__Sequence * array);

/// Create array of msg/PredictedTracks messages.
/**
 * It allocates the memory for the array and calls
 * tracker_interfaces__msg__PredictedTracks__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
tracker_interfaces__msg__PredictedTracks__Sequence *
tracker_interfaces__msg__PredictedTracks__Sequence__create(size_t size);

/// Destroy array of msg/PredictedTracks messages.
/**
 * It calls
 * tracker_interfaces__msg__PredictedTracks__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
void
tracker_interfaces__msg__PredictedTracks__Sequence__destroy(tracker_interfaces__msg__PredictedTracks__Sequence * array);

/// Check for msg/PredictedTracks message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__Sequence__are_equal(const tracker_interfaces__msg__PredictedTracks__Sequence * lhs, const tracker_interfaces__msg__PredictedTracks__Sequence * rhs);

/// Copy an array of msg/PredictedTracks messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_tracker_interfaces
bool
tracker_interfaces__msg__PredictedTracks__Sequence__copy(
  const tracker_interfaces__msg__PredictedTracks__Sequence * input,
  tracker_interfaces__msg__PredictedTracks__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__FUNCTIONS_H_
