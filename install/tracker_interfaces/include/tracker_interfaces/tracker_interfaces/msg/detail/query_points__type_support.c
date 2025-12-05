// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tracker_interfaces:msg/QueryPoints.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tracker_interfaces/msg/detail/query_points__rosidl_typesupport_introspection_c.h"
#include "tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tracker_interfaces/msg/detail/query_points__functions.h"
#include "tracker_interfaces/msg/detail/query_points__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `points`
#include "geometry_msgs/msg/point.h"
// Member `points`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tracker_interfaces__msg__QueryPoints__init(message_memory);
}

void tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_fini_function(void * message_memory)
{
  tracker_interfaces__msg__QueryPoints__fini(message_memory);
}

size_t tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__size_function__QueryPoints__points(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_const_function__QueryPoints__points(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_function__QueryPoints__points(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__fetch_function__QueryPoints__points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_const_function__QueryPoints__points(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__assign_function__QueryPoints__points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_function__QueryPoints__points(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__resize_function__QueryPoints__points(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__QueryPoints, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__QueryPoints, points),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__size_function__QueryPoints__points,  // size() function pointer
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_const_function__QueryPoints__points,  // get_const(index) function pointer
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__get_function__QueryPoints__points,  // get(index) function pointer
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__fetch_function__QueryPoints__points,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__assign_function__QueryPoints__points,  // assign(index, value) function pointer
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__resize_function__QueryPoints__points  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_members = {
  "tracker_interfaces__msg",  // message namespace
  "QueryPoints",  // message name
  2,  // number of fields
  sizeof(tracker_interfaces__msg__QueryPoints),
  tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_member_array,  // message members
  tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_init_function,  // function to initialize message memory (memory has to be allocated)
  tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_type_support_handle = {
  0,
  &tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tracker_interfaces, msg, QueryPoints)() {
  tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_type_support_handle.typesupport_identifier) {
    tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tracker_interfaces__msg__QueryPoints__rosidl_typesupport_introspection_c__QueryPoints_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
