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

#ifndef NOVA_EXAMPLE__TALKER_HPP_
#define NOVA_EXAMPLE__TALKER_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

namespace nova::example
{
class Talker : public rclcpp::Node
{
public:
  explicit Talker(const rclcpp::NodeOptions & options);

private:
  void callback();

  image_transport::Publisher publisher_;  //!< Publisher for compressed video messages
  rclcpp::TimerBase::SharedPtr timer_;    //!< Timer for publishing compressed video messages
};
}  // namespace nova::example
#endif  // NOVA_EXAMPLE__TALKER_HPP_
