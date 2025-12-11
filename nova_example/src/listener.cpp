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

#include "nova_example/listener.hpp"

#include <image_transport/image_transport.hpp>
#include <image_transport/transport_hints.hpp>

namespace nova::example
{
Listener::Listener(const rclcpp::NodeOptions & options) : Node("listener", options)
{
  // set TransportHints to "nova"
  subscriber_ = image_transport::create_subscription(
    this, "nova/image",
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) { this->callback(msg); }, "nova",
    rclcpp::QoS(1).get_rmw_qos_profile());
}

void Listener::callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  RCLCPP_INFO(
    get_logger(), "Received Image %ux%u encoding=%s", msg->width, msg->height,
    msg->encoding.c_str());
}
}  // namespace nova::example

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(nova::example::Listener)
