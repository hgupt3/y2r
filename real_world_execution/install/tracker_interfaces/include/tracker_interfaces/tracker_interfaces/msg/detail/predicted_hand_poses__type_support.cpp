// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tracker_interfaces/msg/detail/predicted_hand_poses__struct.hpp"
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

void PredictedHandPoses_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tracker_interfaces::msg::PredictedHandPoses(_init);
}

void PredictedHandPoses_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tracker_interfaces::msg::PredictedHandPoses *>(message_memory);
  typed_message->~PredictedHandPoses();
}

size_t size_function__PredictedHandPoses__left_trajectory_u(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__left_trajectory_u(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__left_trajectory_u(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__left_trajectory_u(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__left_trajectory_u(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__left_trajectory_u(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__left_trajectory_u(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__left_trajectory_u(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__left_trajectory_v(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__left_trajectory_v(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__left_trajectory_v(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__left_trajectory_v(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__left_trajectory_v(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__left_trajectory_v(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__left_trajectory_v(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__left_trajectory_v(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__left_trajectory_d(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__left_trajectory_d(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__left_trajectory_d(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__left_trajectory_d(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__left_trajectory_d(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__left_trajectory_d(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__left_trajectory_d(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__left_trajectory_d(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__left_final_rotation(const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * get_const_function__PredictedHandPoses__left_final_rotation(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 9> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__left_final_rotation(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 9> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__left_final_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__left_final_rotation(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__left_final_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__left_final_rotation(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

size_t size_function__PredictedHandPoses__right_trajectory_u(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__right_trajectory_u(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__right_trajectory_u(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__right_trajectory_u(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__right_trajectory_u(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__right_trajectory_u(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__right_trajectory_u(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__right_trajectory_u(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__right_trajectory_v(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__right_trajectory_v(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__right_trajectory_v(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__right_trajectory_v(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__right_trajectory_v(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__right_trajectory_v(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__right_trajectory_v(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__right_trajectory_v(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__right_trajectory_d(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PredictedHandPoses__right_trajectory_d(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__right_trajectory_d(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__right_trajectory_d(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__right_trajectory_d(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__right_trajectory_d(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__right_trajectory_d(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PredictedHandPoses__right_trajectory_d(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PredictedHandPoses__right_final_rotation(const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * get_const_function__PredictedHandPoses__right_final_rotation(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 9> *>(untyped_member);
  return &member[index];
}

void * get_function__PredictedHandPoses__right_final_rotation(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 9> *>(untyped_member);
  return &member[index];
}

void fetch_function__PredictedHandPoses__right_final_rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PredictedHandPoses__right_final_rotation(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PredictedHandPoses__right_final_rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PredictedHandPoses__right_final_rotation(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PredictedHandPoses_message_member_array[12] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, header),  // bytes offset in struct
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
    offsetof(tracker_interfaces::msg::PredictedHandPoses, num_timesteps),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "left_valid",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, left_valid),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "left_trajectory_u",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, left_trajectory_u),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__left_trajectory_u,  // size() function pointer
    get_const_function__PredictedHandPoses__left_trajectory_u,  // get_const(index) function pointer
    get_function__PredictedHandPoses__left_trajectory_u,  // get(index) function pointer
    fetch_function__PredictedHandPoses__left_trajectory_u,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__left_trajectory_u,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__left_trajectory_u  // resize(index) function pointer
  },
  {
    "left_trajectory_v",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, left_trajectory_v),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__left_trajectory_v,  // size() function pointer
    get_const_function__PredictedHandPoses__left_trajectory_v,  // get_const(index) function pointer
    get_function__PredictedHandPoses__left_trajectory_v,  // get(index) function pointer
    fetch_function__PredictedHandPoses__left_trajectory_v,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__left_trajectory_v,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__left_trajectory_v  // resize(index) function pointer
  },
  {
    "left_trajectory_d",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, left_trajectory_d),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__left_trajectory_d,  // size() function pointer
    get_const_function__PredictedHandPoses__left_trajectory_d,  // get_const(index) function pointer
    get_function__PredictedHandPoses__left_trajectory_d,  // get(index) function pointer
    fetch_function__PredictedHandPoses__left_trajectory_d,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__left_trajectory_d,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__left_trajectory_d  // resize(index) function pointer
  },
  {
    "left_final_rotation",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, left_final_rotation),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__left_final_rotation,  // size() function pointer
    get_const_function__PredictedHandPoses__left_final_rotation,  // get_const(index) function pointer
    get_function__PredictedHandPoses__left_final_rotation,  // get(index) function pointer
    fetch_function__PredictedHandPoses__left_final_rotation,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__left_final_rotation,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "right_valid",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, right_valid),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "right_trajectory_u",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, right_trajectory_u),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__right_trajectory_u,  // size() function pointer
    get_const_function__PredictedHandPoses__right_trajectory_u,  // get_const(index) function pointer
    get_function__PredictedHandPoses__right_trajectory_u,  // get(index) function pointer
    fetch_function__PredictedHandPoses__right_trajectory_u,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__right_trajectory_u,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__right_trajectory_u  // resize(index) function pointer
  },
  {
    "right_trajectory_v",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, right_trajectory_v),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__right_trajectory_v,  // size() function pointer
    get_const_function__PredictedHandPoses__right_trajectory_v,  // get_const(index) function pointer
    get_function__PredictedHandPoses__right_trajectory_v,  // get(index) function pointer
    fetch_function__PredictedHandPoses__right_trajectory_v,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__right_trajectory_v,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__right_trajectory_v  // resize(index) function pointer
  },
  {
    "right_trajectory_d",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, right_trajectory_d),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__right_trajectory_d,  // size() function pointer
    get_const_function__PredictedHandPoses__right_trajectory_d,  // get_const(index) function pointer
    get_function__PredictedHandPoses__right_trajectory_d,  // get(index) function pointer
    fetch_function__PredictedHandPoses__right_trajectory_d,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__right_trajectory_d,  // assign(index, value) function pointer
    resize_function__PredictedHandPoses__right_trajectory_d  // resize(index) function pointer
  },
  {
    "right_final_rotation",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tracker_interfaces::msg::PredictedHandPoses, right_final_rotation),  // bytes offset in struct
    nullptr,  // default value
    size_function__PredictedHandPoses__right_final_rotation,  // size() function pointer
    get_const_function__PredictedHandPoses__right_final_rotation,  // get_const(index) function pointer
    get_function__PredictedHandPoses__right_final_rotation,  // get(index) function pointer
    fetch_function__PredictedHandPoses__right_final_rotation,  // fetch(index, &value) function pointer
    assign_function__PredictedHandPoses__right_final_rotation,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PredictedHandPoses_message_members = {
  "tracker_interfaces::msg",  // message namespace
  "PredictedHandPoses",  // message name
  12,  // number of fields
  sizeof(tracker_interfaces::msg::PredictedHandPoses),
  PredictedHandPoses_message_member_array,  // message members
  PredictedHandPoses_init_function,  // function to initialize message memory (memory has to be allocated)
  PredictedHandPoses_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PredictedHandPoses_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PredictedHandPoses_message_members,
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
get_message_type_support_handle<tracker_interfaces::msg::PredictedHandPoses>()
{
  return &::tracker_interfaces::msg::rosidl_typesupport_introspection_cpp::PredictedHandPoses_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tracker_interfaces, msg, PredictedHandPoses)() {
  return &::tracker_interfaces::msg::rosidl_typesupport_introspection_cpp::PredictedHandPoses_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
