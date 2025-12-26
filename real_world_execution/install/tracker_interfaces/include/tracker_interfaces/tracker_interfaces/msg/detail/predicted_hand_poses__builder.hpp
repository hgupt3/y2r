// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__BUILDER_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/msg/detail/predicted_hand_poses__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace msg
{

namespace builder
{

class Init_PredictedHandPoses_right_final_rotation
{
public:
  explicit Init_PredictedHandPoses_right_final_rotation(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::msg::PredictedHandPoses right_final_rotation(::tracker_interfaces::msg::PredictedHandPoses::_right_final_rotation_type arg)
  {
    msg_.right_final_rotation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_right_trajectory_d
{
public:
  explicit Init_PredictedHandPoses_right_trajectory_d(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_right_final_rotation right_trajectory_d(::tracker_interfaces::msg::PredictedHandPoses::_right_trajectory_d_type arg)
  {
    msg_.right_trajectory_d = std::move(arg);
    return Init_PredictedHandPoses_right_final_rotation(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_right_trajectory_v
{
public:
  explicit Init_PredictedHandPoses_right_trajectory_v(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_right_trajectory_d right_trajectory_v(::tracker_interfaces::msg::PredictedHandPoses::_right_trajectory_v_type arg)
  {
    msg_.right_trajectory_v = std::move(arg);
    return Init_PredictedHandPoses_right_trajectory_d(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_right_trajectory_u
{
public:
  explicit Init_PredictedHandPoses_right_trajectory_u(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_right_trajectory_v right_trajectory_u(::tracker_interfaces::msg::PredictedHandPoses::_right_trajectory_u_type arg)
  {
    msg_.right_trajectory_u = std::move(arg);
    return Init_PredictedHandPoses_right_trajectory_v(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_right_valid
{
public:
  explicit Init_PredictedHandPoses_right_valid(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_right_trajectory_u right_valid(::tracker_interfaces::msg::PredictedHandPoses::_right_valid_type arg)
  {
    msg_.right_valid = std::move(arg);
    return Init_PredictedHandPoses_right_trajectory_u(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_left_final_rotation
{
public:
  explicit Init_PredictedHandPoses_left_final_rotation(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_right_valid left_final_rotation(::tracker_interfaces::msg::PredictedHandPoses::_left_final_rotation_type arg)
  {
    msg_.left_final_rotation = std::move(arg);
    return Init_PredictedHandPoses_right_valid(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_left_trajectory_d
{
public:
  explicit Init_PredictedHandPoses_left_trajectory_d(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_left_final_rotation left_trajectory_d(::tracker_interfaces::msg::PredictedHandPoses::_left_trajectory_d_type arg)
  {
    msg_.left_trajectory_d = std::move(arg);
    return Init_PredictedHandPoses_left_final_rotation(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_left_trajectory_v
{
public:
  explicit Init_PredictedHandPoses_left_trajectory_v(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_left_trajectory_d left_trajectory_v(::tracker_interfaces::msg::PredictedHandPoses::_left_trajectory_v_type arg)
  {
    msg_.left_trajectory_v = std::move(arg);
    return Init_PredictedHandPoses_left_trajectory_d(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_left_trajectory_u
{
public:
  explicit Init_PredictedHandPoses_left_trajectory_u(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_left_trajectory_v left_trajectory_u(::tracker_interfaces::msg::PredictedHandPoses::_left_trajectory_u_type arg)
  {
    msg_.left_trajectory_u = std::move(arg);
    return Init_PredictedHandPoses_left_trajectory_v(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_left_valid
{
public:
  explicit Init_PredictedHandPoses_left_valid(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_left_trajectory_u left_valid(::tracker_interfaces::msg::PredictedHandPoses::_left_valid_type arg)
  {
    msg_.left_valid = std::move(arg);
    return Init_PredictedHandPoses_left_trajectory_u(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_num_timesteps
{
public:
  explicit Init_PredictedHandPoses_num_timesteps(::tracker_interfaces::msg::PredictedHandPoses & msg)
  : msg_(msg)
  {}
  Init_PredictedHandPoses_left_valid num_timesteps(::tracker_interfaces::msg::PredictedHandPoses::_num_timesteps_type arg)
  {
    msg_.num_timesteps = std::move(arg);
    return Init_PredictedHandPoses_left_valid(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

class Init_PredictedHandPoses_header
{
public:
  Init_PredictedHandPoses_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PredictedHandPoses_num_timesteps header(::tracker_interfaces::msg::PredictedHandPoses::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PredictedHandPoses_num_timesteps(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedHandPoses msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::msg::PredictedHandPoses>()
{
  return tracker_interfaces::msg::builder::Init_PredictedHandPoses_header();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__BUILDER_HPP_
