// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tracker_interfaces:msg/PredictedHandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_HPP_

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
# define DEPRECATED__tracker_interfaces__msg__PredictedHandPoses __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__msg__PredictedHandPoses __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PredictedHandPoses_
{
  using Type = PredictedHandPoses_<ContainerAllocator>;

  explicit PredictedHandPoses_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_timesteps = 0l;
      this->left_valid = false;
      std::fill<typename std::array<float, 9>::iterator, float>(this->left_final_rotation.begin(), this->left_final_rotation.end(), 0.0f);
      this->right_valid = false;
      std::fill<typename std::array<float, 9>::iterator, float>(this->right_final_rotation.begin(), this->right_final_rotation.end(), 0.0f);
    }
  }

  explicit PredictedHandPoses_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    left_final_rotation(_alloc),
    right_final_rotation(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_timesteps = 0l;
      this->left_valid = false;
      std::fill<typename std::array<float, 9>::iterator, float>(this->left_final_rotation.begin(), this->left_final_rotation.end(), 0.0f);
      this->right_valid = false;
      std::fill<typename std::array<float, 9>::iterator, float>(this->right_final_rotation.begin(), this->right_final_rotation.end(), 0.0f);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _num_timesteps_type =
    int32_t;
  _num_timesteps_type num_timesteps;
  using _left_valid_type =
    bool;
  _left_valid_type left_valid;
  using _left_trajectory_u_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _left_trajectory_u_type left_trajectory_u;
  using _left_trajectory_v_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _left_trajectory_v_type left_trajectory_v;
  using _left_trajectory_d_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _left_trajectory_d_type left_trajectory_d;
  using _left_final_rotation_type =
    std::array<float, 9>;
  _left_final_rotation_type left_final_rotation;
  using _right_valid_type =
    bool;
  _right_valid_type right_valid;
  using _right_trajectory_u_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _right_trajectory_u_type right_trajectory_u;
  using _right_trajectory_v_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _right_trajectory_v_type right_trajectory_v;
  using _right_trajectory_d_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _right_trajectory_d_type right_trajectory_d;
  using _right_final_rotation_type =
    std::array<float, 9>;
  _right_final_rotation_type right_final_rotation;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__num_timesteps(
    const int32_t & _arg)
  {
    this->num_timesteps = _arg;
    return *this;
  }
  Type & set__left_valid(
    const bool & _arg)
  {
    this->left_valid = _arg;
    return *this;
  }
  Type & set__left_trajectory_u(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->left_trajectory_u = _arg;
    return *this;
  }
  Type & set__left_trajectory_v(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->left_trajectory_v = _arg;
    return *this;
  }
  Type & set__left_trajectory_d(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->left_trajectory_d = _arg;
    return *this;
  }
  Type & set__left_final_rotation(
    const std::array<float, 9> & _arg)
  {
    this->left_final_rotation = _arg;
    return *this;
  }
  Type & set__right_valid(
    const bool & _arg)
  {
    this->right_valid = _arg;
    return *this;
  }
  Type & set__right_trajectory_u(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->right_trajectory_u = _arg;
    return *this;
  }
  Type & set__right_trajectory_v(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->right_trajectory_v = _arg;
    return *this;
  }
  Type & set__right_trajectory_d(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->right_trajectory_d = _arg;
    return *this;
  }
  Type & set__right_final_rotation(
    const std::array<float, 9> & _arg)
  {
    this->right_final_rotation = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__msg__PredictedHandPoses
    std::shared_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__msg__PredictedHandPoses
    std::shared_ptr<tracker_interfaces::msg::PredictedHandPoses_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PredictedHandPoses_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->num_timesteps != other.num_timesteps) {
      return false;
    }
    if (this->left_valid != other.left_valid) {
      return false;
    }
    if (this->left_trajectory_u != other.left_trajectory_u) {
      return false;
    }
    if (this->left_trajectory_v != other.left_trajectory_v) {
      return false;
    }
    if (this->left_trajectory_d != other.left_trajectory_d) {
      return false;
    }
    if (this->left_final_rotation != other.left_final_rotation) {
      return false;
    }
    if (this->right_valid != other.right_valid) {
      return false;
    }
    if (this->right_trajectory_u != other.right_trajectory_u) {
      return false;
    }
    if (this->right_trajectory_v != other.right_trajectory_v) {
      return false;
    }
    if (this->right_trajectory_d != other.right_trajectory_d) {
      return false;
    }
    if (this->right_final_rotation != other.right_final_rotation) {
      return false;
    }
    return true;
  }
  bool operator!=(const PredictedHandPoses_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PredictedHandPoses_

// alias to use template instance with default allocator
using PredictedHandPoses =
  tracker_interfaces::msg::PredictedHandPoses_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__PREDICTED_HAND_POSES__STRUCT_HPP_
