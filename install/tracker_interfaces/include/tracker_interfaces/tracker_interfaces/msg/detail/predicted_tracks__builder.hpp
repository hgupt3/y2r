// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__BUILDER_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/msg/detail/predicted_tracks__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace msg
{

namespace builder
{

class Init_PredictedTracks_num_timesteps
{
public:
  explicit Init_PredictedTracks_num_timesteps(::tracker_interfaces::msg::PredictedTracks & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::msg::PredictedTracks num_timesteps(::tracker_interfaces::msg::PredictedTracks::_num_timesteps_type arg)
  {
    msg_.num_timesteps = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

class Init_PredictedTracks_num_points
{
public:
  explicit Init_PredictedTracks_num_points(::tracker_interfaces::msg::PredictedTracks & msg)
  : msg_(msg)
  {}
  Init_PredictedTracks_num_timesteps num_points(::tracker_interfaces::msg::PredictedTracks::_num_points_type arg)
  {
    msg_.num_points = std::move(arg);
    return Init_PredictedTracks_num_timesteps(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

class Init_PredictedTracks_trajectory_z
{
public:
  explicit Init_PredictedTracks_trajectory_z(::tracker_interfaces::msg::PredictedTracks & msg)
  : msg_(msg)
  {}
  Init_PredictedTracks_num_points trajectory_z(::tracker_interfaces::msg::PredictedTracks::_trajectory_z_type arg)
  {
    msg_.trajectory_z = std::move(arg);
    return Init_PredictedTracks_num_points(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

class Init_PredictedTracks_trajectory_y
{
public:
  explicit Init_PredictedTracks_trajectory_y(::tracker_interfaces::msg::PredictedTracks & msg)
  : msg_(msg)
  {}
  Init_PredictedTracks_trajectory_z trajectory_y(::tracker_interfaces::msg::PredictedTracks::_trajectory_y_type arg)
  {
    msg_.trajectory_y = std::move(arg);
    return Init_PredictedTracks_trajectory_z(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

class Init_PredictedTracks_trajectory_x
{
public:
  explicit Init_PredictedTracks_trajectory_x(::tracker_interfaces::msg::PredictedTracks & msg)
  : msg_(msg)
  {}
  Init_PredictedTracks_trajectory_y trajectory_x(::tracker_interfaces::msg::PredictedTracks::_trajectory_x_type arg)
  {
    msg_.trajectory_x = std::move(arg);
    return Init_PredictedTracks_trajectory_y(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

class Init_PredictedTracks_header
{
public:
  Init_PredictedTracks_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PredictedTracks_trajectory_x header(::tracker_interfaces::msg::PredictedTracks::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PredictedTracks_trajectory_x(msg_);
  }

private:
  ::tracker_interfaces::msg::PredictedTracks msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::msg::PredictedTracks>()
{
  return tracker_interfaces::msg::builder::Init_PredictedTracks_header();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__BUILDER_HPP_
