// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__TRAITS_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tracker_interfaces/msg/detail/hand_poses__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tracker_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const HandPoses & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: left_u
  {
    out << "left_u: ";
    rosidl_generator_traits::value_to_yaml(msg.left_u, out);
    out << ", ";
  }

  // member: left_v
  {
    out << "left_v: ";
    rosidl_generator_traits::value_to_yaml(msg.left_v, out);
    out << ", ";
  }

  // member: left_depth
  {
    out << "left_depth: ";
    rosidl_generator_traits::value_to_yaml(msg.left_depth, out);
    out << ", ";
  }

  // member: left_rotation
  {
    if (msg.left_rotation.size() == 0) {
      out << "left_rotation: []";
    } else {
      out << "left_rotation: [";
      size_t pending_items = msg.left_rotation.size();
      for (auto item : msg.left_rotation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_valid
  {
    out << "left_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.left_valid, out);
    out << ", ";
  }

  // member: right_u
  {
    out << "right_u: ";
    rosidl_generator_traits::value_to_yaml(msg.right_u, out);
    out << ", ";
  }

  // member: right_v
  {
    out << "right_v: ";
    rosidl_generator_traits::value_to_yaml(msg.right_v, out);
    out << ", ";
  }

  // member: right_depth
  {
    out << "right_depth: ";
    rosidl_generator_traits::value_to_yaml(msg.right_depth, out);
    out << ", ";
  }

  // member: right_rotation
  {
    if (msg.right_rotation.size() == 0) {
      out << "right_rotation: []";
    } else {
      out << "right_rotation: [";
      size_t pending_items = msg.right_rotation.size();
      for (auto item : msg.right_rotation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_valid
  {
    out << "right_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.right_valid, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const HandPoses & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: left_u
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_u: ";
    rosidl_generator_traits::value_to_yaml(msg.left_u, out);
    out << "\n";
  }

  // member: left_v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_v: ";
    rosidl_generator_traits::value_to_yaml(msg.left_v, out);
    out << "\n";
  }

  // member: left_depth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_depth: ";
    rosidl_generator_traits::value_to_yaml(msg.left_depth, out);
    out << "\n";
  }

  // member: left_rotation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_rotation.size() == 0) {
      out << "left_rotation: []\n";
    } else {
      out << "left_rotation:\n";
      for (auto item : msg.left_rotation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_valid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.left_valid, out);
    out << "\n";
  }

  // member: right_u
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_u: ";
    rosidl_generator_traits::value_to_yaml(msg.right_u, out);
    out << "\n";
  }

  // member: right_v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_v: ";
    rosidl_generator_traits::value_to_yaml(msg.right_v, out);
    out << "\n";
  }

  // member: right_depth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_depth: ";
    rosidl_generator_traits::value_to_yaml(msg.right_depth, out);
    out << "\n";
  }

  // member: right_rotation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_rotation.size() == 0) {
      out << "right_rotation: []\n";
    } else {
      out << "right_rotation:\n";
      for (auto item : msg.right_rotation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_valid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.right_valid, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const HandPoses & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace tracker_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use tracker_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tracker_interfaces::msg::HandPoses & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::msg::HandPoses & msg)
{
  return tracker_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::msg::HandPoses>()
{
  return "tracker_interfaces::msg::HandPoses";
}

template<>
inline const char * name<tracker_interfaces::msg::HandPoses>()
{
  return "tracker_interfaces/msg/HandPoses";
}

template<>
struct has_fixed_size<tracker_interfaces::msg::HandPoses>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tracker_interfaces::msg::HandPoses>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tracker_interfaces::msg::HandPoses>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__TRAITS_HPP_
