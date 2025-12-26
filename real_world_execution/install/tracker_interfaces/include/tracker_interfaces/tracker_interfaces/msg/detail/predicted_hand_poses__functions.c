// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice
#include "tracker_interfaces/msg/detail/predicted_hand_poses__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `left_trajectory_u`
// Member `left_trajectory_v`
// Member `left_trajectory_d`
// Member `right_trajectory_u`
// Member `right_trajectory_v`
// Member `right_trajectory_d`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
tracker_interfaces__msg__PredictedHandPoses__init(tracker_interfaces__msg__PredictedHandPoses * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // num_timesteps
  // left_valid
  // left_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__init(&msg->left_trajectory_u, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // left_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__init(&msg->left_trajectory_v, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // left_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__init(&msg->left_trajectory_d, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // left_final_rotation
  // right_valid
  // right_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__init(&msg->right_trajectory_u, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // right_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__init(&msg->right_trajectory_v, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // right_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__init(&msg->right_trajectory_d, 0)) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
    return false;
  }
  // right_final_rotation
  return true;
}

void
tracker_interfaces__msg__PredictedHandPoses__fini(tracker_interfaces__msg__PredictedHandPoses * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // num_timesteps
  // left_valid
  // left_trajectory_u
  rosidl_runtime_c__float__Sequence__fini(&msg->left_trajectory_u);
  // left_trajectory_v
  rosidl_runtime_c__float__Sequence__fini(&msg->left_trajectory_v);
  // left_trajectory_d
  rosidl_runtime_c__float__Sequence__fini(&msg->left_trajectory_d);
  // left_final_rotation
  // right_valid
  // right_trajectory_u
  rosidl_runtime_c__float__Sequence__fini(&msg->right_trajectory_u);
  // right_trajectory_v
  rosidl_runtime_c__float__Sequence__fini(&msg->right_trajectory_v);
  // right_trajectory_d
  rosidl_runtime_c__float__Sequence__fini(&msg->right_trajectory_d);
  // right_final_rotation
}

bool
tracker_interfaces__msg__PredictedHandPoses__are_equal(const tracker_interfaces__msg__PredictedHandPoses * lhs, const tracker_interfaces__msg__PredictedHandPoses * rhs)
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
  // num_timesteps
  if (lhs->num_timesteps != rhs->num_timesteps) {
    return false;
  }
  // left_valid
  if (lhs->left_valid != rhs->left_valid) {
    return false;
  }
  // left_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->left_trajectory_u), &(rhs->left_trajectory_u)))
  {
    return false;
  }
  // left_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->left_trajectory_v), &(rhs->left_trajectory_v)))
  {
    return false;
  }
  // left_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->left_trajectory_d), &(rhs->left_trajectory_d)))
  {
    return false;
  }
  // left_final_rotation
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->left_final_rotation[i] != rhs->left_final_rotation[i]) {
      return false;
    }
  }
  // right_valid
  if (lhs->right_valid != rhs->right_valid) {
    return false;
  }
  // right_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->right_trajectory_u), &(rhs->right_trajectory_u)))
  {
    return false;
  }
  // right_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->right_trajectory_v), &(rhs->right_trajectory_v)))
  {
    return false;
  }
  // right_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->right_trajectory_d), &(rhs->right_trajectory_d)))
  {
    return false;
  }
  // right_final_rotation
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->right_final_rotation[i] != rhs->right_final_rotation[i]) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__msg__PredictedHandPoses__copy(
  const tracker_interfaces__msg__PredictedHandPoses * input,
  tracker_interfaces__msg__PredictedHandPoses * output)
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
  // num_timesteps
  output->num_timesteps = input->num_timesteps;
  // left_valid
  output->left_valid = input->left_valid;
  // left_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->left_trajectory_u), &(output->left_trajectory_u)))
  {
    return false;
  }
  // left_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->left_trajectory_v), &(output->left_trajectory_v)))
  {
    return false;
  }
  // left_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->left_trajectory_d), &(output->left_trajectory_d)))
  {
    return false;
  }
  // left_final_rotation
  for (size_t i = 0; i < 9; ++i) {
    output->left_final_rotation[i] = input->left_final_rotation[i];
  }
  // right_valid
  output->right_valid = input->right_valid;
  // right_trajectory_u
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->right_trajectory_u), &(output->right_trajectory_u)))
  {
    return false;
  }
  // right_trajectory_v
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->right_trajectory_v), &(output->right_trajectory_v)))
  {
    return false;
  }
  // right_trajectory_d
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->right_trajectory_d), &(output->right_trajectory_d)))
  {
    return false;
  }
  // right_final_rotation
  for (size_t i = 0; i < 9; ++i) {
    output->right_final_rotation[i] = input->right_final_rotation[i];
  }
  return true;
}

tracker_interfaces__msg__PredictedHandPoses *
tracker_interfaces__msg__PredictedHandPoses__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedHandPoses * msg = (tracker_interfaces__msg__PredictedHandPoses *)allocator.allocate(sizeof(tracker_interfaces__msg__PredictedHandPoses), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tracker_interfaces__msg__PredictedHandPoses));
  bool success = tracker_interfaces__msg__PredictedHandPoses__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tracker_interfaces__msg__PredictedHandPoses__destroy(tracker_interfaces__msg__PredictedHandPoses * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tracker_interfaces__msg__PredictedHandPoses__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tracker_interfaces__msg__PredictedHandPoses__Sequence__init(tracker_interfaces__msg__PredictedHandPoses__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedHandPoses * data = NULL;

  if (size) {
    data = (tracker_interfaces__msg__PredictedHandPoses *)allocator.zero_allocate(size, sizeof(tracker_interfaces__msg__PredictedHandPoses), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tracker_interfaces__msg__PredictedHandPoses__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tracker_interfaces__msg__PredictedHandPoses__fini(&data[i - 1]);
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
tracker_interfaces__msg__PredictedHandPoses__Sequence__fini(tracker_interfaces__msg__PredictedHandPoses__Sequence * array)
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
      tracker_interfaces__msg__PredictedHandPoses__fini(&array->data[i]);
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

tracker_interfaces__msg__PredictedHandPoses__Sequence *
tracker_interfaces__msg__PredictedHandPoses__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__PredictedHandPoses__Sequence * array = (tracker_interfaces__msg__PredictedHandPoses__Sequence *)allocator.allocate(sizeof(tracker_interfaces__msg__PredictedHandPoses__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tracker_interfaces__msg__PredictedHandPoses__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tracker_interfaces__msg__PredictedHandPoses__Sequence__destroy(tracker_interfaces__msg__PredictedHandPoses__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tracker_interfaces__msg__PredictedHandPoses__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tracker_interfaces__msg__PredictedHandPoses__Sequence__are_equal(const tracker_interfaces__msg__PredictedHandPoses__Sequence * lhs, const tracker_interfaces__msg__PredictedHandPoses__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tracker_interfaces__msg__PredictedHandPoses__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__msg__PredictedHandPoses__Sequence__copy(
  const tracker_interfaces__msg__PredictedHandPoses__Sequence * input,
  tracker_interfaces__msg__PredictedHandPoses__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tracker_interfaces__msg__PredictedHandPoses);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tracker_interfaces__msg__PredictedHandPoses * data =
      (tracker_interfaces__msg__PredictedHandPoses *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tracker_interfaces__msg__PredictedHandPoses__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tracker_interfaces__msg__PredictedHandPoses__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tracker_interfaces__msg__PredictedHandPoses__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
