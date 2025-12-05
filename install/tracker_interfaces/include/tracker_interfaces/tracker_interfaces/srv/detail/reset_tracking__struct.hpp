// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tracker_interfaces:srv/ResetTracking.idl
// generated code does not contain a copyright notice

#ifndef TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_HPP_
#define TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__tracker_interfaces__srv__ResetTracking_Request __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__srv__ResetTracking_Request __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ResetTracking_Request_
{
  using Type = ResetTracking_Request_<ContainerAllocator>;

  explicit ResetTracking_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit ResetTracking_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__srv__ResetTracking_Request
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__srv__ResetTracking_Request
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ResetTracking_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const ResetTracking_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ResetTracking_Request_

// alias to use template instance with default allocator
using ResetTracking_Request =
  tracker_interfaces::srv::ResetTracking_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tracker_interfaces


#ifndef _WIN32
# define DEPRECATED__tracker_interfaces__srv__ResetTracking_Response __attribute__((deprecated))
#else
# define DEPRECATED__tracker_interfaces__srv__ResetTracking_Response __declspec(deprecated)
#endif

namespace tracker_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ResetTracking_Response_
{
  using Type = ResetTracking_Response_<ContainerAllocator>;

  explicit ResetTracking_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  explicit ResetTracking_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tracker_interfaces__srv__ResetTracking_Response
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tracker_interfaces__srv__ResetTracking_Response
    std::shared_ptr<tracker_interfaces::srv::ResetTracking_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ResetTracking_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const ResetTracking_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ResetTracking_Response_

// alias to use template instance with default allocator
using ResetTracking_Response =
  tracker_interfaces::srv::ResetTracking_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tracker_interfaces

namespace tracker_interfaces
{

namespace srv
{

struct ResetTracking
{
  using Request = tracker_interfaces::srv::ResetTracking_Request;
  using Response = tracker_interfaces::srv::ResetTracking_Response;
};

}  // namespace srv

}  // namespace tracker_interfaces

#endif  // TRACKER_INTERFACES__SRV__DETAIL__RESET_TRACKING__STRUCT_HPP_
