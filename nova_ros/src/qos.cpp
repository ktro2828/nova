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

#include "nova_ros/qos.hpp"

#include <string>

namespace nova::ros
{
std::optional<rclcpp::QoS> find_qos(rclcpp::Node * node, const std::string & topic_name)
{
  const auto qos_list = node->get_publishers_info_by_topic(topic_name);
  if (qos_list.size() < 1) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Waiting for topic: " << topic_name << " ...");
    return std::nullopt;
  } else if (qos_list.size() > 1) {
    RCLCPP_ERROR_STREAM(
      node->get_logger(),
      "Multiple publishers found for topic: " << topic_name << ". Cannot determine proper QoS");
    return std::nullopt;
  } else {
    RCLCPP_INFO_STREAM(node->get_logger(), "QoS is acquired for topic: " << topic_name);
    return qos_list[0].qos_profile();
  }
}
}  // namespace nova::ros
