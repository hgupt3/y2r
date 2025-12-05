// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tracker_interfaces:msg/QueryPoints.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_HPP_
#define TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_HPP_

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
// Member 'points'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tracker_interfaces__msg__QueryPoints __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__msg__QueryPoints __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct QueryPoints_
{
  using Type = QueryPoints_<ContainerAllocator>;

  explicit QueryPoints_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit QueryPoints_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _points_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>>;
  _points_type points;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__points(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>> & _arg)
  {
    this->points = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::msg::QueryPoints_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::msg::QueryPoints_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::QueryPoints_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::msg::QueryPoints_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__msg__QueryPoints
    std::shared_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__msg__QueryPoints
    std::shared_ptr<tracker_interfaces::msg::QueryPoints_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const QueryPoints_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->points != other.points) {
      return false;
    }
    return true;
  }
  bool operator!=(const QueryPoints_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct QueryPoints_

// alias to use template instance with default allocator
using QueryPoints =
  tracker_interfaces::msg::QueryPoints_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__MSG__DETAIL__QUERY_POINTS__STRUCT_HPP_
