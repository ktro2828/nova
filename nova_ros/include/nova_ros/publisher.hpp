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

#ifndef NOVA_ROS__PUBLISHER_HPP_
#define NOVA_ROS__PUBLISHER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace nova::ros
{
using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;

class Publisher : public rclcpp::Node
{
public:
  explicit Publisher(const rclcpp::NodeOptions & options);
  ~Publisher();

private:
  void on_image(Image::ConstSharedPtr msg);
  void on_camera_info(CameraInfo::ConstSharedPtr msg);

  rclcpp::Subscription<Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_subscription_;
};
}  // namespace nova::ros
#endif  // NOVA_ROS__PUBLISHER_HPP_
