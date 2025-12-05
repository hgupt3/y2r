// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tracker_interfaces:srv/ResetTracking.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__TRAITS_HPP_
#define TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tracker_interfaces/srv/detail/reset_tracking__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const ResetTracking_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ResetTracking_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ResetTracking_Request & msg, bool use_flow_style = false)
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
  const tracker_interfaces::srv::ResetTracking_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::srv::ResetTracking_Request & msg)
{
  return tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::srv::ResetTracking_Request>()
{
  return "tracker_interfaces::srv::ResetTracking_Request";
}

template<>
inline const char * name<tracker_interfaces::srv::ResetTracking_Request>()
{
  return "tracker_interfaces/srv/ResetTracking_Request";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::ResetTracking_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<tracker_interfaces::srv::ResetTracking_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<tracker_interfaces::srv::ResetTracking_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const ResetTracking_Response & msg,
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
  const ResetTracking_Response & msg,
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

inline std::string to_yaml(const ResetTracking_Response & msg, bool use_flow_style = false)
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
  const tracker_interfaces::srv::ResetTracking_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::srv::ResetTracking_Response & msg)
{
  return tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::srv::ResetTracking_Response>()
{
  return "tracker_interfaces::srv::ResetTracking_Response";
}

template<>
inline const char * name<tracker_interfaces::srv::ResetTracking_Response>()
{
  return "tracker_interfaces/srv/ResetTracking_Response";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::ResetTracking_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tracker_interfaces::srv::ResetTracking_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tracker_interfaces::srv::ResetTracking_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tracker_interfaces::srv::ResetTracking>()
{
  return "tracker_interfaces::srv::ResetTracking";
}

template<>
inline const char * name<tracker_interfaces::srv::ResetTracking>()
{
  return "tracker_interfaces/srv/ResetTracking";
}

template<>
struct has_fixed_size<tracker_interfaces::srv::ResetTracking>
  : std::integral_constant<
    bool,
    has_fixed_size<tracker_interfaces::srv::ResetTracking_Request>::value &&
    has_fixed_size<tracker_interfaces::srv::ResetTracking_Response>::value
  >
{
};

template<>
struct has_bounded_size<tracker_interfaces::srv::ResetTracking>
  : std::integral_constant<
    bool,
    has_bounded_size<tracker_interfaces::srv::ResetTracking_Request>::value &&
    has_bounded_size<tracker_interfaces::srv::ResetTracking_Response>::value
  >
{
};

template<>
struct is_service<tracker_interfaces::srv::ResetTracking>
  : std::true_type
{
};

template<>
struct is_service_request<tracker_interfaces::srv::ResetTracking_Request>
  : std::true_type
{
};

template<>
struct is_service_response<tracker_interfaces::srv::ResetTracking_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__TRAITS_HPP_
