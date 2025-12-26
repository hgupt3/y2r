// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tracker_interfaces:srv/SetPredictionPrompt.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__BUILDER_HPP_
#define TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tracker_interfaces/srv/detail/set_prediction_prompt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tracker_interfaces
{

namespace srv
{

namespace builder
{

class Init_SetPredictionPrompt_Request_prompt
{
public:
  Init_SetPredictionPrompt_Request_prompt()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::tracker_interfaces::srv::SetPredictionPrompt_Request prompt(::tracker_interfaces::srv::SetPredictionPrompt_Request::_prompt_type arg)
  {
    msg_.prompt = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::srv::SetPredictionPrompt_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::srv::SetPredictionPrompt_Request>()
{
  return tracker_interfaces::srv::builder::Init_SetPredictionPrompt_Request_prompt();
}

}  // namespace tracker_interfaces


namespace tracker_interfaces
{

namespace srv
{

namespace builder
{

class Init_SetPredictionPrompt_Response_message
{
public:
  explicit Init_SetPredictionPrompt_Response_message(::tracker_interfaces::srv::SetPredictionPrompt_Response & msg)
  : msg_(msg)
  {}
  ::tracker_interfaces::srv::SetPredictionPrompt_Response message(::tracker_interfaces::srv::SetPredictionPrompt_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tracker_interfaces::srv::SetPredictionPrompt_Response msg_;
};

class Init_SetPredictionPrompt_Response_success
{
public:
  Init_SetPredictionPrompt_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetPredictionPrompt_Response_message success(::tracker_interfaces::srv::SetPredictionPrompt_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetPredictionPrompt_Response_message(msg_);
  }

private:
  ::tracker_interfaces::srv::SetPredictionPrompt_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tracker_interfaces::srv::SetPredictionPrompt_Response>()
{
  return tracker_interfaces::srv::builder::Init_SetPredictionPrompt_Response_success();
}

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__BUILDER_HPP_
