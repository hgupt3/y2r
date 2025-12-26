// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tracker_interfaces:msg/HandPoses.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_HPP_

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
# define DEPRECATED__tracker_interfaces__msg__HandPoses __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__msg__HandPoses __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct HandPoses_
{
  using Type = HandPoses_<ContainerAllocator>;

  explicit HandPoses_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_u = 0.0f;
      this->left_v = 0.0f;
      this->left_depth = 0.0f;
      std::fill<typename std::array<float, 9>::iterator, float>(this->left_rotation.begin(), this->left_rotation.end(), 0.0f);
      this->left_valid = false;
      this->right_u = 0.0f;
      this->right_v = 0.0f;
      this->right_depth = 0.0f;
      std::fill<typename std::array<float, 9>::iterator, float>(this->right_rotation.begin(), this->right_rotation.end(), 0.0f);
      this->right_valid = false;
    }
  }

  explicit HandPoses_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    left_rotation(_alloc),
    right_rotation(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->left_u = 0.0f;
      this->left_v = 0.0f;
      this->left_depth = 0.0f;
      std::fill<typename std::array<float, 9>::iterator, float>(this->left_rotation.begin(), this->left_rotation.end(), 0.0f);
      this->left_valid = false;
      this->right_u = 0.0f;
      this->right_v = 0.0f;
      this->right_depth = 0.0f;
      std::fill<typename std::array<float, 9>::iterator, float>(this->right_rotation.begin(), this->right_rotation.end(), 0.0f);
      this->right_valid = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _left_u_type =
    float;
  _left_u_type left_u;
  using _left_v_type =
    float;
  _left_v_type left_v;
  using _left_depth_type =
    float;
  _left_depth_type left_depth;
  using _left_rotation_type =
    std::array<float, 9>;
  _left_rotation_type left_rotation;
  using _left_valid_type =
    bool;
  _left_valid_type left_valid;
  using _right_u_type =
    float;
  _right_u_type right_u;
  using _right_v_type =
    float;
  _right_v_type right_v;
  using _right_depth_type =
    float;
  _right_depth_type right_depth;
  using _right_rotation_type =
    std::array<float, 9>;
  _right_rotation_type right_rotation;
  using _right_valid_type =
    bool;
  _right_valid_type right_valid;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__left_u(
    const float & _arg)
  {
    this->left_u = _arg;
    return *this;
  }
  Type & set__left_v(
    const float & _arg)
  {
    this->left_v = _arg;
    return *this;
  }
  Type & set__left_depth(
    const float & _arg)
  {
    this->left_depth = _arg;
    return *this;
  }
  Type & set__left_rotation(
    const std::array<float, 9> & _arg)
  {
    this->left_rotation = _arg;
    return *this;
  }
  Type & set__left_valid(
    const bool & _arg)
  {
    this->left_valid = _arg;
    return *this;
  }
  Type & set__right_u(
    const float & _arg)
  {
    this->right_u = _arg;
    return *this;
  }
  Type & set__right_v(
    const float & _arg)
  {
    this->right_v = _arg;
    return *this;
  }
  Type & set__right_depth(
    const float & _arg)
  {
    this->right_depth = _arg;
    return *this;
  }
  Type & set__right_rotation(
    const std::array<float, 9> & _arg)
  {
    this->right_rotation = _arg;
    return *this;
  }
  Type & set__right_valid(
    const bool & _arg)
  {
    this->right_valid = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::msg::HandPoses_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::msg::HandPoses_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::HandPoses_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::HandPoses_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__msg__HandPoses
    std::shared_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__msg__HandPoses
    std::shared_ptr<tracker_interfaces::msg::HandPoses_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const HandPoses_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->left_u != other.left_u) {
      return false;
    }
    if (this->left_v != other.left_v) {
      return false;
    }
    if (this->left_depth != other.left_depth) {
      return false;
    }
    if (this->left_rotation != other.left_rotation) {
      return false;
    }
    if (this->left_valid != other.left_valid) {
      return false;
    }
    if (this->right_u != other.right_u) {
      return false;
    }
    if (this->right_v != other.right_v) {
      return false;
    }
    if (this->right_depth != other.right_depth) {
      return false;
    }
    if (this->right_rotation != other.right_rotation) {
      return false;
    }
    if (this->right_valid != other.right_valid) {
      return false;
    }
    return true;
  }
  bool operator!=(const HandPoses_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct HandPoses_

// alias to use template instance with default allocator
using HandPoses =
  tracker_interfaces::msg::HandPoses_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__HAND_POSES__STRUCT_HPP_
