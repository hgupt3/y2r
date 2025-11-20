// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice
#include "tracker_interfaces/msg/detail/predicted_tracks__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `query_points`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `trajectory_x`
// Member `trajectory_y`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
tracker_interfaces__msg__PredictedTracks__init(tracker_interfaces__msg__PredictedTracks * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tracker_interfaces__msg__PredictedTracks__fini(msg);
    return false;
  }
  // query_points
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->query_points, 0)) {
    tracker_interfaces__msg__PredictedTracks__fini(msg);
    return false;
  }
  // trajectory_x
  if (!rosidl_runtime_c__float__Sequence__init(&msg->trajectory_x, 0)) {
    tracker_interfaces__msg__PredictedTracks__fini(msg);
    return false;
  }
  // trajectory_y
  if (!rosidl_runtime_c__float__Sequence__init(&msg->trajectory_y, 0)) {
    tracker_interfaces__msg__PredictedTracks__fini(msg);
    return false;
  }
  // num_points
  // num_timesteps
  return true;
}

void
tracker_interfaces__msg__PredictedTracks__fini(tracker_interfaces__msg__PredictedTracks * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // query_points
  geometry_msgs__msg__Point__Sequence__fini(&msg->query_points);
  // trajectory_x
  rosidl_runtime_c__float__Sequence__fini(&msg->trajectory_x);
  // trajectory_y
  rosidl_runtime_c__float__Sequence__fini(&msg->trajectory_y);
  // num_points
  // num_timesteps
}

bool
tracker_interfaces__msg__PredictedTracks__are_equal(const tracker_interfaces__msg__PredictedTracks * lhs, const tracker_interfaces__msg__PredictedTracks * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // query_points
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->query_points), &(rhs->query_points)))
  {
    return false;
  }
  // trajectory_x
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->trajectory_x), &(rhs->trajectory_x)))
  {
    return false;
  }
  // trajectory_y
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->trajectory_y), &(rhs->trajectory_y)))
  {
    return false;
  }
  // num_points
  if (lhs->num_points != rhs->num_points) {
    return false;
  }
  // num_timesteps
  if (lhs->num_timesteps != rhs->num_timesteps) {
    return false;
  }
  return true;
}

bool
tracker_interfaces__msg__PredictedTracks__copy(
  const tracker_interfaces__msg__PredictedTracks * input,
  tracker_interfaces__msg__PredictedTracks * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // query_points
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->query_points), &(output->query_points)))
  {
    return false;
  }
  // trajectory_x
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->trajectory_x), &(output->trajectory_x)))
  {
    return false;
  }
  // trajectory_y
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->trajectory_y), &(output->trajectory_y)))
  {
    return false;
  }
  // num_points
  output->num_points = input->num_points;
  // num_timesteps
  output->num_timesteps = input->num_timesteps;
  return true;
}

tracker_interfaces__msg__PredictedTracks *
tracker_interfaces__msg__PredictedTracks__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedTracks * msg = (tracker_interfaces__msg__PredictedTracks *)allocator.allocate(sizeof(tracker_interfaces__msg__PredictedTracks), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tracker_interfaces__msg__PredictedTracks));
  bool success = tracker_interfaces__msg__PredictedTracks__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tracker_interfaces__msg__PredictedTracks__destroy(tracker_interfaces__msg__PredictedTracks * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tracker_interfaces__msg__PredictedTracks__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tracker_interfaces__msg__PredictedTracks__Sequence__init(tracker_interfaces__msg__PredictedTracks__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedTracks * data = NULL;

  if (size) {
    data = (tracker_interfaces__msg__PredictedTracks *)allocator.zero_allocate(size, sizeof(tracker_interfaces__msg__PredictedTracks), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tracker_interfaces__msg__PredictedTracks__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tracker_interfaces__msg__PredictedTracks__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
tracker_interfaces__msg__PredictedTracks__Sequence__fini(tracker_interfaces__msg__PredictedTracks__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      tracker_interfaces__msg__PredictedTracks__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

tracker_interfaces__msg__PredictedTracks__Sequence *
tracker_interfaces__msg__PredictedTracks__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedTracks__Sequence * array = (tracker_interfaces__msg__PredictedTracks__Sequence *)allocator.allocate(sizeof(tracker_interfaces__msg__PredictedTracks__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tracker_interfaces__msg__PredictedTracks__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tracker_interfaces__msg__PredictedTracks__Sequence__destroy(tracker_interfaces__msg__PredictedTracks__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tracker_interfaces__msg__PredictedTracks__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tracker_interfaces__msg__PredictedTracks__Sequence__are_equal(const tracker_interfaces__msg__PredictedTracks__Sequence * lhs, const tracker_interfaces__msg__PredictedTracks__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tracker_interfaces__msg__PredictedTracks__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__msg__PredictedTracks__Sequence__copy(
  const tracker_interfaces__msg__PredictedTracks__Sequence * input,
  tracker_interfaces__msg__PredictedTracks__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tracker_interfaces__msg__PredictedTracks);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tracker_interfaces__msg__PredictedTracks * data =
      (tracker_interfaces__msg__PredictedTracks *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tracker_interfaces__msg__PredictedTracks__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tracker_interfaces__msg__PredictedTracks__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tracker_interfaces__msg__PredictedTracks__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
