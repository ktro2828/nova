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

#ifndef NOVA_ROS__COMPRESSED_VIDEO_PUBLISHER_HPP_
#define NOVA_ROS__COMPRESSED_VIDEO_PUBLISHER_HPP_

#include "nova_ros/parameter_definition.hpp"

#include <ffmpeg_encoder_decoder/encoder.hpp>
#include <image_transport/simple_publisher_plugin.hpp>
#include <nova_common/datatype.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nova_msgs/msg/compressed_video.hpp>

#include <string>

namespace nova::ros
{
using CompressedVideo = nova_msgs::msg::CompressedVideo;

class CompressedVideoPublisher : public image_transport::SimplePublisherPlugin<CompressedVideo>
{
public:
#ifdef IMAGE_TRANSPORT_USE_PUBLISHER_T
  using PublisherTFn = PublisherT;
#else
  using PublisherTFn = PublishFn;
#endif
#ifdef IMAGE_TRANSPORT_USE_QOS
  using QoSType = rclcpp::QoS;
#else
  using QoSType = rmw_qos_profile_t;
#endif
#ifdef IMAGE_TRANSPORT_USE_NODEINTERFACE
  using NodeType = image_transport::RequiredInterfaces;
#else
  using NodeType = rclcpp::Node *;
#endif

  CompressedVideoPublisher();
  ~CompressedVideoPublisher() override;
  std::string getTransportName() const override { return "compressedVideo"; }

protected:
#ifdef IMAGE_TRANSPORT_NEEDS_PUBLISHEROPTIONS
  void advertiseImpl(
    NodeType node, const std::string & base_topic, QoSType custom_qos,
    rclcpp::PublisherOptions opt) override;
#else
  void advertiseImpl(NodeType node, const std::string & base_topic, QoSType custom_qos) override;
#endif
  void publish(const Image & message, const PublisherTFn & publish_fn) const override;

private:
  void callback(
    const std::string & frame_id, const rclcpp::Time & stamp, const std::string & codec,
    uint32_t width, uint32_t height, uint64_t pts, uint8_t flags, uint8_t * data, size_t sz);

  QoSType initialize(NodeType node, const std::string & base_name, QoSType custom_qos);
  void declare_parameter(NodeType node, const ParameterDefinition & definition);
  void handle_av_options(const std::string & option);

  const PublisherTFn * publish_function_{nullptr};
  ffmpeg_encoder_decoder::Encoder encoder_;
  uint32_t frame_count_{0};
  std::string parameter_namespace_;
};
}  // namespace nova::ros
#endif  // NOVA_ROS__COMPRESSED_VIDEO_PUBLISHER_HPP_
