// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__TRAITS_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tracker_interfaces/msg/detail/predicted_hand_poses__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tracker_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PredictedHandPoses & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: num_timesteps
  {
    out << "num_timesteps: ";
    rosidl_generator_traits::value_to_yaml(msg.num_timesteps, out);
    out << ", ";
  }

  // member: left_valid
  {
    out << "left_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.left_valid, out);
    out << ", ";
  }

  // member: left_trajectory_u
  {
    if (msg.left_trajectory_u.size() == 0) {
      out << "left_trajectory_u: []";
    } else {
      out << "left_trajectory_u: [";
      size_t pending_items = msg.left_trajectory_u.size();
      for (auto item : msg.left_trajectory_u) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_trajectory_v
  {
    if (msg.left_trajectory_v.size() == 0) {
      out << "left_trajectory_v: []";
    } else {
      out << "left_trajectory_v: [";
      size_t pending_items = msg.left_trajectory_v.size();
      for (auto item : msg.left_trajectory_v) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_trajectory_d
  {
    if (msg.left_trajectory_d.size() == 0) {
      out << "left_trajectory_d: []";
    } else {
      out << "left_trajectory_d: [";
      size_t pending_items = msg.left_trajectory_d.size();
      for (auto item : msg.left_trajectory_d) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: left_final_rotation
  {
    if (msg.left_final_rotation.size() == 0) {
      out << "left_final_rotation: []";
    } else {
      out << "left_final_rotation: [";
      size_t pending_items = msg.left_final_rotation.size();
      for (auto item : msg.left_final_rotation) {
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
    out << ", ";
  }

  // member: right_trajectory_u
  {
    if (msg.right_trajectory_u.size() == 0) {
      out << "right_trajectory_u: []";
    } else {
      out << "right_trajectory_u: [";
      size_t pending_items = msg.right_trajectory_u.size();
      for (auto item : msg.right_trajectory_u) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_trajectory_v
  {
    if (msg.right_trajectory_v.size() == 0) {
      out << "right_trajectory_v: []";
    } else {
      out << "right_trajectory_v: [";
      size_t pending_items = msg.right_trajectory_v.size();
      for (auto item : msg.right_trajectory_v) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_trajectory_d
  {
    if (msg.right_trajectory_d.size() == 0) {
      out << "right_trajectory_d: []";
    } else {
      out << "right_trajectory_d: [";
      size_t pending_items = msg.right_trajectory_d.size();
      for (auto item : msg.right_trajectory_d) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_final_rotation
  {
    if (msg.right_final_rotation.size() == 0) {
      out << "right_final_rotation: []";
    } else {
      out << "right_final_rotation: [";
      size_t pending_items = msg.right_final_rotation.size();
      for (auto item : msg.right_final_rotation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PredictedHandPoses & msg,
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

  // member: num_timesteps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_timesteps: ";
    rosidl_generator_traits::value_to_yaml(msg.num_timesteps, out);
    out << "\n";
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

  // member: left_trajectory_u
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_trajectory_u.size() == 0) {
      out << "left_trajectory_u: []\n";
    } else {
      out << "left_trajectory_u:\n";
      for (auto item : msg.left_trajectory_u) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_trajectory_v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_trajectory_v.size() == 0) {
      out << "left_trajectory_v: []\n";
    } else {
      out << "left_trajectory_v:\n";
      for (auto item : msg.left_trajectory_v) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_trajectory_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_trajectory_d.size() == 0) {
      out << "left_trajectory_d: []\n";
    } else {
      out << "left_trajectory_d:\n";
      for (auto item : msg.left_trajectory_d) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: left_final_rotation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_final_rotation.size() == 0) {
      out << "left_final_rotation: []\n";
    } else {
      out << "left_final_rotation:\n";
      for (auto item : msg.left_final_rotation) {
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

  // member: right_trajectory_u
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_trajectory_u.size() == 0) {
      out << "right_trajectory_u: []\n";
    } else {
      out << "right_trajectory_u:\n";
      for (auto item : msg.right_trajectory_u) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_trajectory_v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_trajectory_v.size() == 0) {
      out << "right_trajectory_v: []\n";
    } else {
      out << "right_trajectory_v:\n";
      for (auto item : msg.right_trajectory_v) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_trajectory_d
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_trajectory_d.size() == 0) {
      out << "right_trajectory_d: []\n";
    } else {
      out << "right_trajectory_d:\n";
      for (auto item : msg.right_trajectory_d) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_final_rotation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_final_rotation.size() == 0) {
      out << "right_final_rotation: []\n";
    } else {
      out << "right_final_rotation:\n";
      for (auto item : msg.right_final_rotation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PredictedHandPoses & msg, bool use_flow_style = false)
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
  const tracker_interfaces::msg::PredictedHandPoses & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::msg::PredictedHandPoses & msg)
{
  return tracker_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::msg::PredictedHandPoses>()
{
  return "tracker_interfaces::msg::PredictedHandPoses";
}

template<>
inline const char * name<tracker_interfaces::msg::PredictedHandPoses>()
{
  return "tracker_interfaces/msg/PredictedHandPoses";
}

template<>
struct has_fixed_size<tracker_interfaces::msg::PredictedHandPoses>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tracker_interfaces::msg::PredictedHandPoses>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tracker_interfaces::msg::PredictedHandPoses>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__TRAITS_HPP_
