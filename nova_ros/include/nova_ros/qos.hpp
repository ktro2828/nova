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

#ifndef NOVA_ROS__QOS_HPP_
#define NOVA_ROS__QOS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <optional>
#include <string>

namespace nova::ros
{
/**
 * @brief Get the QoS profile for a given topic.
 *
 * @param node The ROS node to use for querying the topic.
 * @param topic_name The name of the topic to query.
 * @return std::optional<rclcpp::QoS> The QoS profile for the topic, or std::nullopt if no
 * publishers are found or if multiple publishers are found.
 */
std::optional<rclcpp::QoS> find_qos(rclcpp::Node * node, const std::string & topic_name);
}  // namespace nova::ros
#endif  // NOVA_ROS__QOS_HPP_
