// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NOVA_ROS__PARAMETER_DEFINITION_HPP_
#define NOVA_ROS__PARAMETER_DEFINITION_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <string>

namespace nova::ros
{
struct ParameterDefinition
{
  using ParameterDescriptor = rcl_interfaces::msg::ParameterDescriptor;
  using ParameterValue = rclcpp::ParameterValue;

#ifdef IMAGE_TRANSPORT_USE_NODEINTERFACE
  using NodeType = image_transport::RequiredInterfaces;
#else
  using NodeType = rclcpp::Node *;
#endif

  rclcpp::ParameterValue declare_parameter(NodeType node, const std::string & param_base) const;

  ParameterValue default_value;
  ParameterDescriptor descriptor;
  std::string warning_if_not_set;
};
}  // namespace nova::ros
#endif  // NOVA_ROS__PARAMETER_DEFINITION_HPP_
