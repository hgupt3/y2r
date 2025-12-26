// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from tracker_interfaces:srv/SetPrompt.idl
// generated code does not contain a copyright notice
#include "tracker_interfaces/srv/detail/set_prompt__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `prompt`
#include "rosidl_runtime_c/string_functions.h"

bool
tracker_interfaces__srv__SetPrompt_Request__init(tracker_interfaces__srv__SetPrompt_Request * msg)
{
  if (!msg) {
    return false;
  }
  // prompt
  if (!rosidl_runtime_c__String__init(&msg->prompt)) {
    tracker_interfaces__srv__SetPrompt_Request__fini(msg);
    return false;
  }
  return true;
}

void
tracker_interfaces__srv__SetPrompt_Request__fini(tracker_interfaces__srv__SetPrompt_Request * msg)
{
  if (!msg) {
    return;
  }
  // prompt
  rosidl_runtime_c__String__fini(&msg->prompt);
}

bool
tracker_interfaces__srv__SetPrompt_Request__are_equal(const tracker_interfaces__srv__SetPrompt_Request * lhs, const tracker_interfaces__srv__SetPrompt_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // prompt
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->prompt), &(rhs->prompt)))
  {
    return false;
  }
  return true;
}

bool
tracker_interfaces__srv__SetPrompt_Request__copy(
  const tracker_interfaces__srv__SetPrompt_Request * input,
  tracker_interfaces__srv__SetPrompt_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // prompt
  if (!rosidl_runtime_c__String__copy(
      &(input->prompt), &(output->prompt)))
  {
    return false;
  }
  return true;
}

tracker_interfaces__srv__SetPrompt_Request *
tracker_interfaces__srv__SetPrompt_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Request * msg = (tracker_interfaces__srv__SetPrompt_Request *)allocator.allocate(sizeof(tracker_interfaces__srv__SetPrompt_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tracker_interfaces__srv__SetPrompt_Request));
  bool success = tracker_interfaces__srv__SetPrompt_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tracker_interfaces__srv__SetPrompt_Request__destroy(tracker_interfaces__srv__SetPrompt_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tracker_interfaces__srv__SetPrompt_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tracker_interfaces__srv__SetPrompt_Request__Sequence__init(tracker_interfaces__srv__SetPrompt_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Request * data = NULL;

  if (size) {
    data = (tracker_interfaces__srv__SetPrompt_Request *)allocator.zero_allocate(size, sizeof(tracker_interfaces__srv__SetPrompt_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tracker_interfaces__srv__SetPrompt_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tracker_interfaces__srv__SetPrompt_Request__fini(&data[i - 1]);
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
tracker_interfaces__srv__SetPrompt_Request__Sequence__fini(tracker_interfaces__srv__SetPrompt_Request__Sequence * array)
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
      tracker_interfaces__srv__SetPrompt_Request__fini(&array->data[i]);
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

tracker_interfaces__srv__SetPrompt_Request__Sequence *
tracker_interfaces__srv__SetPrompt_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Request__Sequence * array = (tracker_interfaces__srv__SetPrompt_Request__Sequence *)allocator.allocate(sizeof(tracker_interfaces__srv__SetPrompt_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tracker_interfaces__srv__SetPrompt_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tracker_interfaces__srv__SetPrompt_Request__Sequence__destroy(tracker_interfaces__srv__SetPrompt_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tracker_interfaces__srv__SetPrompt_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tracker_interfaces__srv__SetPrompt_Request__Sequence__are_equal(const tracker_interfaces__srv__SetPrompt_Request__Sequence * lhs, const tracker_interfaces__srv__SetPrompt_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tracker_interfaces__srv__SetPrompt_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__srv__SetPrompt_Request__Sequence__copy(
  const tracker_interfaces__srv__SetPrompt_Request__Sequence * input,
  tracker_interfaces__srv__SetPrompt_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tracker_interfaces__srv__SetPrompt_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tracker_interfaces__srv__SetPrompt_Request * data =
      (tracker_interfaces__srv__SetPrompt_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tracker_interfaces__srv__SetPrompt_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tracker_interfaces__srv__SetPrompt_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tracker_interfaces__srv__SetPrompt_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

bool
tracker_interfaces__srv__SetPrompt_Response__init(tracker_interfaces__srv__SetPrompt_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    tracker_interfaces__srv__SetPrompt_Response__fini(msg);
    return false;
  }
  return true;
}

void
tracker_interfaces__srv__SetPrompt_Response__fini(tracker_interfaces__srv__SetPrompt_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
tracker_interfaces__srv__SetPrompt_Response__are_equal(const tracker_interfaces__srv__SetPrompt_Response * lhs, const tracker_interfaces__srv__SetPrompt_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
tracker_interfaces__srv__SetPrompt_Response__copy(
  const tracker_interfaces__srv__SetPrompt_Response * input,
  tracker_interfaces__srv__SetPrompt_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

tracker_interfaces__srv__SetPrompt_Response *
tracker_interfaces__srv__SetPrompt_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Response * msg = (tracker_interfaces__srv__SetPrompt_Response *)allocator.allocate(sizeof(tracker_interfaces__srv__SetPrompt_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(tracker_interfaces__srv__SetPrompt_Response));
  bool success = tracker_interfaces__srv__SetPrompt_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
tracker_interfaces__srv__SetPrompt_Response__destroy(tracker_interfaces__srv__SetPrompt_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    tracker_interfaces__srv__SetPrompt_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
tracker_interfaces__srv__SetPrompt_Response__Sequence__init(tracker_interfaces__srv__SetPrompt_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Response * data = NULL;

  if (size) {
    data = (tracker_interfaces__srv__SetPrompt_Response *)allocator.zero_allocate(size, sizeof(tracker_interfaces__srv__SetPrompt_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = tracker_interfaces__srv__SetPrompt_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        tracker_interfaces__srv__SetPrompt_Response__fini(&data[i - 1]);
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
tracker_interfaces__srv__SetPrompt_Response__Sequence__fini(tracker_interfaces__srv__SetPrompt_Response__Sequence * array)
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
      tracker_interfaces__srv__SetPrompt_Response__fini(&array->data[i]);
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

tracker_interfaces__srv__SetPrompt_Response__Sequence *
tracker_interfaces__srv__SetPrompt_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  tracker_interfaces__srv__SetPrompt_Response__Sequence * array = (tracker_interfaces__srv__SetPrompt_Response__Sequence *)allocator.allocate(sizeof(tracker_interfaces__srv__SetPrompt_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = tracker_interfaces__srv__SetPrompt_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
tracker_interfaces__srv__SetPrompt_Response__Sequence__destroy(tracker_interfaces__srv__SetPrompt_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    tracker_interfaces__srv__SetPrompt_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
tracker_interfaces__srv__SetPrompt_Response__Sequence__are_equal(const tracker_interfaces__srv__SetPrompt_Response__Sequence * lhs, const tracker_interfaces__srv__SetPrompt_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!tracker_interfaces__srv__SetPrompt_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
tracker_interfaces__srv__SetPrompt_Response__Sequence__copy(
  const tracker_interfaces__srv__SetPrompt_Response__Sequence * input,
  tracker_interfaces__srv__SetPrompt_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(tracker_interfaces__srv__SetPrompt_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    tracker_interfaces__srv__SetPrompt_Response * data =
      (tracker_interfaces__srv__SetPrompt_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!tracker_interfaces__srv__SetPrompt_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          tracker_interfaces__srv__SetPrompt_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!tracker_interfaces__srv__SetPrompt_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
