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

#ifndef NOVA_EXAMPLE__LISTENER_HPP_
#define NOVA_EXAMPLE__LISTENER_HPP_

#include <image_transport/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

namespace nova::example
{
class Listener : public rclcpp::Node
{
public:
  explicit Listener(const rclcpp::NodeOptions & options);

private:
  void callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  image_transport::Subscriber subscriber_;  //!< Subscriber for compressed video messages
};
}  // namespace nova::example
#endif  // NOVA_EXAMPLE__LISTENER_HPP_
