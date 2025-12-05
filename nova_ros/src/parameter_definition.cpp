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

#include "nova_ros/parameter_definition.hpp"

#include <string>

namespace nova::ros
{
rclcpp::ParameterValue ParameterDefinition::declare_parameter(
  NodeType node, const std::string & param_base) const
{
#ifdef IMAGE_TRANSPORT_USE_NODEINTERFACE
  auto params = node.get_node_parameters_interface();
  auto logger = node.get_node_logging_interface()->get_logger();
#else
  auto params = node->get_node_parameters_interface();
  auto logger = node->get_node_logging_interface()->get_logger();
#endif

  const std::string param_name = param_base + descriptor.name;
  rclcpp::ParameterValue value;
  try {
    const auto & map = params->get_parameter_overrides();
    if (map.find(param_name) == map.end() && !warning_if_not_set.empty()) {
      RCLCPP_WARN_STREAM(logger, warning_if_not_set);
    }
    value = params->declare_parameter(param_name, default_value, descriptor);
  } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
    RCLCPP_DEBUG_STREAM(logger, "Parameter " << descriptor.name << " already declared");
    value = params->get_parameter(param_name).get_parameter_value();
  }
  return value;
}
}  // namespace nova::ros
