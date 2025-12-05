// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tracker_interfaces:msg/PredictedTracks.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tracker_interfaces__msg__PredictedTracks __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__msg__PredictedTracks __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PredictedTracks_
{
  using Type = PredictedTracks_<ContainerAllocator>;

  explicit PredictedTracks_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_points = 0l;
      this->num_timesteps = 0l;
    }
  }

  explicit PredictedTracks_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_points = 0l;
      this->num_timesteps = 0l;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _trajectory_x_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _trajectory_x_type trajectory_x;
  using _trajectory_y_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _trajectory_y_type trajectory_y;
  using _trajectory_z_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _trajectory_z_type trajectory_z;
  using _num_points_type =
    int32_t;
  _num_points_type num_points;
  using _num_timesteps_type =
    int32_t;
  _num_timesteps_type num_timesteps;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__trajectory_x(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->trajectory_x = _arg;
    return *this;
  }
  Type & set__trajectory_y(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->trajectory_y = _arg;
    return *this;
  }
  Type & set__trajectory_z(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->trajectory_z = _arg;
    return *this;
  }
  Type & set__num_points(
    const int32_t & _arg)
  {
    this->num_points = _arg;
    return *this;
  }
  Type & set__num_timesteps(
    const int32_t & _arg)
  {
    this->num_timesteps = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__msg__PredictedTracks
    std::shared_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__msg__PredictedTracks
    std::shared_ptr<tracker_interfaces::msg::PredictedTracks_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PredictedTracks_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->trajectory_x != other.trajectory_x) {
      return false;
    }
    if (this->trajectory_y != other.trajectory_y) {
      return false;
    }
    if (this->trajectory_z != other.trajectory_z) {
      return false;
    }
    if (this->num_points != other.num_points) {
      return false;
    }
    if (this->num_timesteps != other.num_timesteps) {
      return false;
    }
    return true;
  }
  bool operator!=(const PredictedTracks_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PredictedTracks_

// alias to use template instance with default allocator
using PredictedTracks =
  tracker_interfaces::msg::PredictedTracks_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_TRACKS__STRUCT_HPP_
