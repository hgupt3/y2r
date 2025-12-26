// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tracker_interfaces/msg/detail/hand_poses__rosidl_typesupport_introspection_c.h"
#include "tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tracker_interfaces/msg/detail/hand_poses__functions.h"
#include "tracker_interfaces/msg/detail/hand_poses__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tracker_interfaces__msg__HandPoses__init(message_memory);
}

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_fini_function(void * message_memory)
{
  tracker_interfaces__msg__HandPoses__fini(message_memory);
}

size_t tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__size_function__HandPoses__left_rotation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__left_rotation(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__left_rotation(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__fetch_function__HandPoses__left_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__left_rotation(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__assign_function__HandPoses__left_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__left_rotation(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

size_t tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__size_function__HandPoses__right_rotation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__right_rotation(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__right_rotation(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__fetch_function__HandPoses__right_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__right_rotation(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__assign_function__HandPoses__right_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__right_rotation(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_member_array[11] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_u",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, left_u),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_v",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, left_v),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_depth",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, left_depth),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_rotation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, left_rotation),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__size_function__HandPoses__left_rotation,  // size() function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__left_rotation,  // get_const(index) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__left_rotation,  // get(index) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__fetch_function__HandPoses__left_rotation,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__assign_function__HandPoses__left_rotation,  // assign(index, value) function pointer
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
    offsetof(tracker_interfaces__msg__HandPoses, left_valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_u",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, right_u),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_v",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, right_v),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_depth",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, right_depth),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_rotation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__HandPoses, right_rotation),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__size_function__HandPoses__right_rotation,  // size() function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_const_function__HandPoses__right_rotation,  // get_const(index) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__get_function__HandPoses__right_rotation,  // get(index) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__fetch_function__HandPoses__right_rotation,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__assign_function__HandPoses__right_rotation,  // assign(index, value) function pointer
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
    offsetof(tracker_interfaces__msg__HandPoses, right_valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_members = {
  "tracker_interfaces__msg",  // message namespace
  "HandPoses",  // message name
  11,  // number of fields
  sizeof(tracker_interfaces__msg__HandPoses),
  tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_member_array,  // message members
  tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_init_function,  // function to initialize message memory (memory has to be allocated)
  tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_type_support_handle = {
  0,
  &tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tracker_interfaces, msg, HandPoses)() {
  tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_type_support_handle.typesupport_identifier) {
    tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tracker_interfaces__msg__HandPoses__rosidl_typesupport_introspection_c__HandPoses_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
