// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tracker_interfaces/msg/detail/predicted_hand_poses__rosidl_typesupport_introspection_c.h"
#include "tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tracker_interfaces/msg/detail/predicted_hand_poses__functions.h"
#include "tracker_interfaces/msg/detail/predicted_hand_poses__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `left_trajectory_u`
// Member `left_trajectory_v`
// Member `left_trajectory_d`
// Member `right_trajectory_u`
// Member `right_trajectory_v`
// Member `right_trajectory_d`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tracker_interfaces__msg__PredictedHandPoses__init(message_memory);
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_fini_function(void * message_memory)
{
  tracker_interfaces__msg__PredictedHandPoses__fini(message_memory);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_u(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_u(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_u(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_u(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_u(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_u(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_u(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_u(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_v(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_v(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_v(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_v(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_v(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_v(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_v(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_v(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_d(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_d(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_d(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_d(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_d(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_d(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_d(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_d(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_final_rotation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_final_rotation(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_final_rotation(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_final_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_final_rotation(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_final_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_final_rotation(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_u(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_u(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_u(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_u(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_u(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_u(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_u(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_u(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_v(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_v(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_v(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_v(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_v(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_v(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_v(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_v(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_d(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_d(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_d(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_d(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_d(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_d(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_d(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_d(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_final_rotation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_final_rotation(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_final_rotation(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_final_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_final_rotation(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_final_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_final_rotation(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_member_array[12] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_timesteps",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, num_timesteps),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_valid",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, left_valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_trajectory_u",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, left_trajectory_u),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_u,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_u,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_u,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_u,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_u,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_u  // resize(index) function pointer
  },
  {
    "left_trajectory_v",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, left_trajectory_v),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_v,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_v,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_v,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_v,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_v,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_v  // resize(index) function pointer
  },
  {
    "left_trajectory_d",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, left_trajectory_d),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_trajectory_d,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_trajectory_d,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_trajectory_d,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_trajectory_d,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_trajectory_d,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__left_trajectory_d  // resize(index) function pointer
  },
  {
    "left_final_rotation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, left_final_rotation),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__left_final_rotation,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__left_final_rotation,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__left_final_rotation,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__left_final_rotation,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__left_final_rotation,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_valid",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, right_valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_trajectory_u",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, right_trajectory_u),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_u,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_u,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_u,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_u,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_u,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_u  // resize(index) function pointer
  },
  {
    "right_trajectory_v",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, right_trajectory_v),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_v,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_v,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_v,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_v,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_v,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_v  // resize(index) function pointer
  },
  {
    "right_trajectory_d",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, right_trajectory_d),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_trajectory_d,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_trajectory_d,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_trajectory_d,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_trajectory_d,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_trajectory_d,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__resize_function__PredictedHandPoses__right_trajectory_d  // resize(index) function pointer
  },
  {
    "right_final_rotation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedHandPoses, right_final_rotation),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__size_function__PredictedHandPoses__right_final_rotation,  // size() function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_const_function__PredictedHandPoses__right_final_rotation,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__get_function__PredictedHandPoses__right_final_rotation,  // get(index) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__fetch_function__PredictedHandPoses__right_final_rotation,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__assign_function__PredictedHandPoses__right_final_rotation,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_members = {
  "tracker_interfaces__msg",  // message namespace
  "PredictedHandPoses",  // message name
  12,  // number of fields
  sizeof(tracker_interfaces__msg__PredictedHandPoses),
  tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_member_array,  // message members
  tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_init_function,  // function to initialize message memory (memory has to be allocated)
  tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_type_support_handle = {
  0,
  &tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tracker_interfaces, msg, PredictedHandPoses)() {
  tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_type_support_handle.typesupport_identifier) {
    tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tracker_interfaces__msg__PredictedHandPoses__rosidl_typesupport_introspection_c__PredictedHandPoses_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
