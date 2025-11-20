// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:msg/QueryPoints.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__BUILDER_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/msg/detail/query_points__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace msg
{

namespace builder
{

class Init_QueryPoints_points
{
public:
  explicit Init_QueryPoints_points(::tracker_interfaces::msg::QueryPoints & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::msg::QueryPoints points(::tracker_interfaces::msg::QueryPoints::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::msg::QueryPoints msg_;
};

class Init_QueryPoints_header
{
public:
  Init_QueryPoints_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_QueryPoints_points header(::tracker_interfaces::msg::QueryPoints::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_QueryPoints_points(msg_);
  }

private:
  ::tracker_interfaces::msg::QueryPoints msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::msg::QueryPoints>()
{
  return tracker_interfaces::msg::builder::Init_QueryPoints_header();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__BUILDER_HPP_
