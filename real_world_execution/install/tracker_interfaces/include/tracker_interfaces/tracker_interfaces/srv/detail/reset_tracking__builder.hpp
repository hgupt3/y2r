// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:srv/ResetTracking.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__BUILDER_HPP_
#define TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/srv/detail/reset_tracking__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::srv::ResetTracking_Request>()
{
  return ::tracker_interfaces::srv::ResetTracking_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace tracker_interfaces


namespace tracker_interfaces
{

namespace srv
{

namespace builder
{

class Init_ResetTracking_Response_message
{
public:
  explicit Init_ResetTracking_Response_message(::tracker_interfaces::srv::ResetTracking_Response & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::srv::ResetTracking_Response message(::tracker_interfaces::srv::ResetTracking_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::srv::ResetTracking_Response msg_;
};

class Init_ResetTracking_Response_success
{
public:
  Init_ResetTracking_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ResetTracking_Response_message success(::tracker_interfaces::srv::ResetTracking_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_ResetTracking_Response_message(msg_);
  }

private:
  ::tracker_interfaces::srv::ResetTracking_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::srv::ResetTracking_Response>()
{
  return tracker_interfaces::srv::builder::Init_ResetTracking_Response_success();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__BUILDER_HPP_
