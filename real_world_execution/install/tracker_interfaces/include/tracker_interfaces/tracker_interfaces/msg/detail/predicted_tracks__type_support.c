// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tracker_interfaces/msg/detail/predicted_tracks__rosidl_typesupport_introspection_c.h"
#include "tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tracker_interfaces/msg/detail/predicted_tracks__functions.h"
#include "tracker_interfaces/msg/detail/predicted_tracks__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `query_points`
#include "geometry_msgs/msg/point.h"
// Member `query_points`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"
// Member `trajectory_x`
// Member `trajectory_y`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tracker_interfaces__msg__PredictedTracks__init(message_memory);
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_fini_function(void * message_memory)
{
  tracker_interfaces__msg__PredictedTracks__fini(message_memory);
}

size_t tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__query_points(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__query_points(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__query_points(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__query_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__query_points(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__query_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__query_points(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__query_points(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__trajectory_x(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_x(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_x(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__trajectory_x(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_x(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__trajectory_x(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_x(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__trajectory_x(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__trajectory_y(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_y(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_y(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__trajectory_y(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_y(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__trajectory_y(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_y(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__trajectory_y(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedTracks, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "query_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedTracks, query_points),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__query_points,  // size() function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__query_points,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__query_points,  // get(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__query_points,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__query_points,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__query_points  // resize(index) function pointer
  },
  {
    "trajectory_x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedTracks, trajectory_x),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__trajectory_x,  // size() function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_x,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_x,  // get(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__trajectory_x,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__trajectory_x,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__trajectory_x  // resize(index) function pointer
  },
  {
    "trajectory_y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedTracks, trajectory_y),  // bytes offset in struct
    NULL,  // default value
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__size_function__PredictedTracks__trajectory_y,  // size() function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_const_function__PredictedTracks__trajectory_y,  // get_const(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__get_function__PredictedTracks__trajectory_y,  // get(index) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__fetch_function__PredictedTracks__trajectory_y,  // fetch(index, &value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__assign_function__PredictedTracks__trajectory_y,  // assign(index, value) function pointer
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__resize_function__PredictedTracks__trajectory_y  // resize(index) function pointer
  },
  {
    "num_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces__msg__PredictedTracks, num_points),  // bytes offset in struct
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
    offsetof(tracker_interfaces__msg__PredictedTracks, num_timesteps),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_members = {
  "tracker_interfaces__msg",  // message namespace
  "PredictedTracks",  // message name
  6,  // number of fields
  sizeof(tracker_interfaces__msg__PredictedTracks),
  tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_member_array,  // message members
  tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_init_function,  // function to initialize message memory (memory has to be allocated)
  tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_type_support_handle = {
  0,
  &tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tracker_interfaces, msg, PredictedTracks)() {
  tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_type_support_handle.typesupport_identifier) {
    tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tracker_interfaces__msg__PredictedTracks__rosidl_typesupport_introspection_c__PredictedTracks_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
