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

#include "nova_example/talker.hpp"

#include <image_transport/image_transport.hpp>

#include <chrono>
#include <string>

namespace nova::example
{
Talker::Talker(const rclcpp::NodeOptions & options) : Node("talker", options)
{
  // declare transport type parameter as "nova"
  this->declare_parameter<std::string>("image_transport", "compressed_video");
  publisher_ =
    image_transport::create_publisher(this, "nova/image", rclcpp::QoS(1).get_rmw_qos_profile());

  timer_ = this->create_wall_timer(std::chrono::milliseconds(100), [this]() { this->callback(); });
}

void Talker::callback()
{
  sensor_msgs::msg::Image image;
  image.header.stamp = this->now();
  image.header.frame_id = "camera";
  image.height = 480;
  image.width = 640;
  image.encoding = "bgr8";
  image.step = image.width * 3;
  image.data.resize(image.height * image.step);

  publisher_.publish(image);
}
}  // namespace nova::example

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(nova::example::Talker)
