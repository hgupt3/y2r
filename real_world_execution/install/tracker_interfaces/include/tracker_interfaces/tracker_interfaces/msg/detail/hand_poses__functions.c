// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice
#include "tracker_interfaces/msg/detail/hand_poses__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
tracker_interfaces__msg__HandPoses__init(tracker_interfaces__msg__HandPoses * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    tracker_interfaces__msg__HandPoses__fini(msg);
    return false;
  }
  // left_u
  // left_v
  // left_depth
  // left_rotation
  // left_valid
  // right_u
  // right_v
  // right_depth
  // right_rotation
  // right_valid
  return true;
}

void
tracker_interfaces__msg__HandPoses__fini(tracker_interfaces__msg__HandPoses * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // left_u
  // left_v
  // left_depth
  // left_rotation
  // left_valid
  // right_u
  // right_v
  // right_depth
  // right_rotation
  // right_valid
}

bool
tracker_interfaces__msg__HandPoses__are_equal(const tracker_interfaces__msg__HandPoses * lhs, const tracker_interfaces__msg__HandPoses * rhs)
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
  // left_u
  if (lhs->left_u != rhs->left_u) {
    return false;
  }
  // left_v
  if (lhs->left_v != rhs->left_v) {
    return false;
  }
  // left_depth
  if (lhs->left_depth != rhs->left_depth) {
    return false;
  }
  // left_rotation
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->left_rotation[i] != rhs->left_rotation[i]) {
      return false;
    }
  }
  // left_valid
  if (lhs->left_valid != rhs->left_valid) {
    return false;
  }
  // right_u
  if (lhs->right_u != rhs->right_u) {
    return false;
  }
  // right_v
  if (lhs->right_v != rhs->right_v) {
    return false;
  }
  // right_depth
  if (lhs->right_depth != rhs->right_depth) {
    return false;
  }
  // right_rotation
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->right_rotation[i] != rhs->right_rotation[i]) {
      return false;
    }
  }
  // right_valid
  if (lhs->right_valid != rhs->right_valid) {
    return false;
  }
  return true;
}

bool
tracker_interfaces__msg__HandPoses__copy(
  const tracker_interfaces__msg__HandPoses * input,
  tracker_interfaces__msg__HandPoses * output)
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
  // left_u
  output->left_u = input->left_u;
  // left_v
  output->left_v = input->left_v;
  // left_depth
  output->left_depth = input->left_depth;
  // left_rotation
  for (size_t i = 0; i < 9; ++i) {
    output->left_rotation[i] = input->left_rotation[i];
  }
  // left_valid
  output->left_valid = input->left_valid;
  // right_u
  output->right_u = input->right_u;
  // right_v
  output->right_v = input->right_v;
  // right_depth
  output->right_depth = input->right_depth;
  // right_rotation
  for (size_t i = 0; i < 9; ++i) {
    output->right_rotation[i] = input->right_rotation[i];
  }
  // right_valid
  output->right_valid = input->right_valid;
  return true;
}

tracker_interfaces__msg__HandPoses *
tracker_interfaces__msg__HandPoses__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__HandPoses * msg = (tracker_interfaces__msg__HandPoses *)allocator.allocate(sizeof(tracker_interfaces__msg__HandPoses), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tracker_interfaces__msg__HandPoses));
  bool success = tracker_interfaces__msg__HandPoses__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tracker_interfaces__msg__HandPoses__destroy(tracker_interfaces__msg__HandPoses * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tracker_interfaces__msg__HandPoses__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tracker_interfaces__msg__HandPoses__Sequence__init(tracker_interfaces__msg__HandPoses__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__HandPoses * data = NULL;

  if (size) {
    data = (tracker_interfaces__msg__HandPoses *)allocator.zero_allocate(size, sizeof(tracker_interfaces__msg__HandPoses), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tracker_interfaces__msg__HandPoses__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tracker_interfaces__msg__HandPoses__fini(&data[i - 1]);
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
tracker_interfaces__msg__HandPoses__Sequence__fini(tracker_interfaces__msg__HandPoses__Sequence * array)
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
      tracker_interfaces__msg__HandPoses__fini(&array->data[i]);
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

tracker_interfaces__msg__HandPoses__Sequence *
tracker_interfaces__msg__HandPoses__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__msg__HandPoses__Sequence * array = (tracker_interfaces__msg__HandPoses__Sequence *)allocator.allocate(sizeof(tracker_interfaces__msg__HandPoses__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tracker_interfaces__msg__HandPoses__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tracker_interfaces__msg__HandPoses__Sequence__destroy(tracker_interfaces__msg__HandPoses__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tracker_interfaces__msg__HandPoses__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tracker_interfaces__msg__HandPoses__Sequence__are_equal(const tracker_interfaces__msg__HandPoses__Sequence * lhs, const tracker_interfaces__msg__HandPoses__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tracker_interfaces__msg__HandPoses__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__msg__HandPoses__Sequence__copy(
  const tracker_interfaces__msg__HandPoses__Sequence * input,
  tracker_interfaces__msg__HandPoses__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tracker_interfaces__msg__HandPoses);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tracker_interfaces__msg__HandPoses * data =
      (tracker_interfaces__msg__HandPoses *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tracker_interfaces__msg__HandPoses__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tracker_interfaces__msg__HandPoses__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tracker_interfaces__msg__HandPoses__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
