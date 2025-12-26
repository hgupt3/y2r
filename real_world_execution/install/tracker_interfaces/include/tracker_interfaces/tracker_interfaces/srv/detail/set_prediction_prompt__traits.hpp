// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tracker_interfaces:srv/SetPredictionPrompt.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__TRAITS_HPP_
#define TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tracker_interfaces/srv/detail/set_prediction_prompt__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetPredictionPrompt_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: prompt
  {
    out << "prompt: ";
    rosidl_generator_traits::value_to_yaml(msg.prompt, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetPredictionPrompt_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: prompt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "prompt: ";
    rosidl_generator_traits::value_to_yaml(msg.prompt, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetPredictionPrompt_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace tracker_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use tracker_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tracker_interfaces::srv::SetPredictionPrompt_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::srv::SetPredictionPrompt_Request & msg)
{
  return tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::srv::SetPredictionPrompt_Request>()
{
  return "tracker_interfaces::srv::SetPredictionPrompt_Request";
}

template<>
inline const char * name<tracker_interfaces::srv::SetPredictionPrompt_Request>()
{
  return "tracker_interfaces/srv/SetPredictionPrompt_Request";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::SetPredictionPrompt_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tracker_interfaces::srv::SetPredictionPrompt_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tracker_interfaces::srv::SetPredictionPrompt_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetPredictionPrompt_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetPredictionPrompt_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetPredictionPrompt_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace tracker_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use tracker_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tracker_interfaces::srv::SetPredictionPrompt_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::srv::SetPredictionPrompt_Response & msg)
{
  return tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::srv::SetPredictionPrompt_Response>()
{
  return "tracker_interfaces::srv::SetPredictionPrompt_Response";
}

template<>
inline const char * name<tracker_interfaces::srv::SetPredictionPrompt_Response>()
{
  return "tracker_interfaces/srv/SetPredictionPrompt_Response";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::SetPredictionPrompt_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tracker_interfaces::srv::SetPredictionPrompt_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tracker_interfaces::srv::SetPredictionPrompt_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tracker_interfaces::srv::SetPredictionPrompt>()
{
  return "tracker_interfaces::srv::SetPredictionPrompt";
}

template<>
inline const char * name<tracker_interfaces::srv::SetPredictionPrompt>()
{
  return "tracker_interfaces/srv/SetPredictionPrompt";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::SetPredictionPrompt>
  : std::integral_constant<
    bool,
    has_fixed_size<tracker_interfaces::srv::SetPredictionPrompt_Request>::value &&
    has_fixed_size<tracker_interfaces::srv::SetPredictionPrompt_Response>::value
  >
{
};

template<>
struct has_bounded_size<tracker_interfaces::srv::SetPredictionPrompt>
  : std::integral_constant<
    bool,
    has_bounded_size<tracker_interfaces::srv::SetPredictionPrompt_Request>::value &&
    has_bounded_size<tracker_interfaces::srv::SetPredictionPrompt_Response>::value
  >
{
};

template<>
struct is_service<tracker_interfaces::srv::SetPredictionPrompt>
  : std::true_type
{
};

template<>
struct is_service_request<tracker_interfaces::srv::SetPredictionPrompt_Request>
  : std::true_type
{
};

template<>
struct is_service_response<tracker_interfaces::srv::SetPredictionPrompt_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TRACKER_INTERFACES__SRV__DETAIL__SET_PREDICTION_PROMPT__TRAITS_HPP_
