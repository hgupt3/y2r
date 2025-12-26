// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__BUILDER_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/msg/detail/hand_poses__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace msg
{

namespace builder
{

class Init_HandPoses_right_valid
{
public:
  explicit Init_HandPoses_right_valid(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::msg::HandPoses right_valid(::tracker_interfaces::msg::HandPoses::_right_valid_type arg)
  {
    msg_.right_valid = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_right_rotation
{
public:
  explicit Init_HandPoses_right_rotation(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_right_valid right_rotation(::tracker_interfaces::msg::HandPoses::_right_rotation_type arg)
  {
    msg_.right_rotation = std::move(arg);
    return Init_HandPoses_right_valid(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_right_depth
{
public:
  explicit Init_HandPoses_right_depth(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_right_rotation right_depth(::tracker_interfaces::msg::HandPoses::_right_depth_type arg)
  {
    msg_.right_depth = std::move(arg);
    return Init_HandPoses_right_rotation(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_right_v
{
public:
  explicit Init_HandPoses_right_v(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_right_depth right_v(::tracker_interfaces::msg::HandPoses::_right_v_type arg)
  {
    msg_.right_v = std::move(arg);
    return Init_HandPoses_right_depth(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_right_u
{
public:
  explicit Init_HandPoses_right_u(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_right_v right_u(::tracker_interfaces::msg::HandPoses::_right_u_type arg)
  {
    msg_.right_u = std::move(arg);
    return Init_HandPoses_right_v(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_left_valid
{
public:
  explicit Init_HandPoses_left_valid(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_right_u left_valid(::tracker_interfaces::msg::HandPoses::_left_valid_type arg)
  {
    msg_.left_valid = std::move(arg);
    return Init_HandPoses_right_u(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_left_rotation
{
public:
  explicit Init_HandPoses_left_rotation(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_left_valid left_rotation(::tracker_interfaces::msg::HandPoses::_left_rotation_type arg)
  {
    msg_.left_rotation = std::move(arg);
    return Init_HandPoses_left_valid(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_left_depth
{
public:
  explicit Init_HandPoses_left_depth(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_left_rotation left_depth(::tracker_interfaces::msg::HandPoses::_left_depth_type arg)
  {
    msg_.left_depth = std::move(arg);
    return Init_HandPoses_left_rotation(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_left_v
{
public:
  explicit Init_HandPoses_left_v(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_left_depth left_v(::tracker_interfaces::msg::HandPoses::_left_v_type arg)
  {
    msg_.left_v = std::move(arg);
    return Init_HandPoses_left_depth(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_left_u
{
public:
  explicit Init_HandPoses_left_u(::tracker_interfaces::msg::HandPoses & msg)
  : msg_(msg)
  {}
  Init_HandPoses_left_v left_u(::tracker_interfaces::msg::HandPoses::_left_u_type arg)
  {
    msg_.left_u = std::move(arg);
    return Init_HandPoses_left_v(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

class Init_HandPoses_header
{
public:
  Init_HandPoses_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_HandPoses_left_u header(::tracker_interfaces::msg::HandPoses::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_HandPoses_left_u(msg_);
  }

private:
  ::tracker_interfaces::msg::HandPoses msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::msg::HandPoses>()
{
  return tracker_interfaces::msg::builder::Init_HandPoses_header();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__BUILDER_HPP_
