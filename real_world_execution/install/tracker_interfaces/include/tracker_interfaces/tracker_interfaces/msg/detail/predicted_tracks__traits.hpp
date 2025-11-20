// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__TRAITS_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tracker_interfaces/msg/detail/predicted_tracks__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'query_points'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace tracker_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PredictedTracks & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: query_points
  {
    if (msg.query_points.size() == 0) {
      out << "query_points: []";
    } else {
      out << "query_points: [";
      size_t pending_items = msg.query_points.size();
      for (auto item : msg.query_points) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: trajectory_x
  {
    if (msg.trajectory_x.size() == 0) {
      out << "trajectory_x: []";
    } else {
      out << "trajectory_x: [";
      size_t pending_items = msg.trajectory_x.size();
      for (auto item : msg.trajectory_x) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: trajectory_y
  {
    if (msg.trajectory_y.size() == 0) {
      out << "trajectory_y: []";
    } else {
      out << "trajectory_y: [";
      size_t pending_items = msg.trajectory_y.size();
      for (auto item : msg.trajectory_y) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: num_points
  {
    out << "num_points: ";
    rosidl_generator_traits::value_to_yaml(msg.num_points, out);
    out << ", ";
  }

  // member: num_timesteps
  {
    out << "num_timesteps: ";
    rosidl_generator_traits::value_to_yaml(msg.num_timesteps, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PredictedTracks & msg,
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

  // member: query_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.query_points.size() == 0) {
      out << "query_points: []\n";
    } else {
      out << "query_points:\n";
      for (auto item : msg.query_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: trajectory_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.trajectory_x.size() == 0) {
      out << "trajectory_x: []\n";
    } else {
      out << "trajectory_x:\n";
      for (auto item : msg.trajectory_x) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: trajectory_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.trajectory_y.size() == 0) {
      out << "trajectory_y: []\n";
    } else {
      out << "trajectory_y:\n";
      for (auto item : msg.trajectory_y) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: num_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_points: ";
    rosidl_generator_traits::value_to_yaml(msg.num_points, out);
    out << "\n";
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PredictedTracks & msg, bool use_flow_style = false)
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
  const tracker_interfaces::msg::PredictedTracks & msg,
  std::ostream & out, size_t indentation = 0)
{
  tracker_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tracker_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const tracker_interfaces::msg::PredictedTracks & msg)
{
  return tracker_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tracker_interfaces::msg::PredictedTracks>()
{
  return "tracker_interfaces::msg::PredictedTracks";
}

template<>
inline const char * name<tracker_interfaces::msg::PredictedTracks>()
{
  return "tracker_interfaces/msg/PredictedTracks";
}

template<>
struct has_fixed_size<tracker_interfaces::msg::PredictedTracks>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tracker_interfaces::msg::PredictedTracks>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tracker_interfaces::msg::PredictedTracks>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__TRAITS_HPP_
