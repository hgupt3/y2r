// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tracker_interfaces/msg/detail/predicted_tracks__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace tracker_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void PredictedTracks_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tracker_interfaces::msg::PredictedTracks(_init);
}

void PredictedTracks_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tracker_interfaces::msg::PredictedTracks *>(message_memory);
  typed_message->~PredictedTracks();
}

size_t size_function__PredictedTracks__query_points(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedTracks__query_points(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedTracks__query_points(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedTracks__query_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const geometry_msgs::msg::Point *>(
    get_const_function__PredictedTracks__query_points(untyped_member, index));
  auto & value = *reinterpret_cast<geometry_msgs::msg::Point *>(untyped_value);
  value = item;
}

void assign_function__PredictedTracks__query_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<geometry_msgs::msg::Point *>(
    get_function__PredictedTracks__query_points(untyped_member, index));
  const auto & value = *reinterpret_cast<const geometry_msgs::msg::Point *>(untyped_value);
  item = value;
}

void resize_function__PredictedTracks__query_points(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedTracks__trajectory_x(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedTracks__trajectory_x(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedTracks__trajectory_x(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedTracks__trajectory_x(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedTracks__trajectory_x(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedTracks__trajectory_x(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedTracks__trajectory_x(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedTracks__trajectory_x(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedTracks__trajectory_y(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedTracks__trajectory_y(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedTracks__trajectory_y(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedTracks__trajectory_y(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedTracks__trajectory_y(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedTracks__trajectory_y(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedTracks__trajectory_y(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedTracks__trajectory_y(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PredictedTracks_message_member_array[6] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "query_points",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, query_points),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedTracks__query_points,  // size() function pointer
    get_const_function__PredictedTracks__query_points,  // get_const(index) function pointer
    get_function__PredictedTracks__query_points,  // get(index) function pointer
    fetch_function__PredictedTracks__query_points,  // fetch(index, &value) function pointer
    assign_function__PredictedTracks__query_points,  // assign(index, value) function pointer
    resize_function__PredictedTracks__query_points  // resize(index) function pointer
  },
  {
    "trajectory_x",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, trajectory_x),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedTracks__trajectory_x,  // size() function pointer
    get_const_function__PredictedTracks__trajectory_x,  // get_const(index) function pointer
    get_function__PredictedTracks__trajectory_x,  // get(index) function pointer
    fetch_function__PredictedTracks__trajectory_x,  // fetch(index, &value) function pointer
    assign_function__PredictedTracks__trajectory_x,  // assign(index, value) function pointer
    resize_function__PredictedTracks__trajectory_x  // resize(index) function pointer
  },
  {
    "trajectory_y",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, trajectory_y),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedTracks__trajectory_y,  // size() function pointer
    get_const_function__PredictedTracks__trajectory_y,  // get_const(index) function pointer
    get_function__PredictedTracks__trajectory_y,  // get(index) function pointer
    fetch_function__PredictedTracks__trajectory_y,  // fetch(index, &value) function pointer
    assign_function__PredictedTracks__trajectory_y,  // assign(index, value) function pointer
    resize_function__PredictedTracks__trajectory_y  // resize(index) function pointer
  },
  {
    "num_points",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, num_points),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "num_timesteps",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedTracks, num_timesteps),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PredictedTracks_message_members = {
  "tracker_interfaces::msg",  // message namespace
  "PredictedTracks",  // message name
  6,  // number of fields
  sizeof(tracker_interfaces::msg::PredictedTracks),
  PredictedTracks_message_member_array,  // message members
  PredictedTracks_init_function,  // function to initialize message memory (memory has to be allocated)
  PredictedTracks_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PredictedTracks_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PredictedTracks_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tracker_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tracker_interfaces::msg::PredictedTracks>()
{
  return &::tracker_interfaces::msg::rosidl_typesupport_introspection_cpp::PredictedTracks_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tracker_interfaces, msg, PredictedTracks)() {
  return &::tracker_interfaces::msg::rosidl_typesupport_introspection_cpp::PredictedTracks_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
