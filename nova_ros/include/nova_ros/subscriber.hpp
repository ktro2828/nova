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

#ifndef NOVA_ROS__SUBSCRIBER_HPP_
#define NOVA_ROS__SUBSCRIBER_HPP_

#include <ffmpeg_encoder_decoder/decoder.hpp>
#include <image_transport/simple_subscriber_plugin.hpp>
#include <nova_common/datatype.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nova_msgs/msg/compressed_video.hpp>

#include <string>

namespace nova::ros
{
using CompressedVideo = nova_msgs::msg::CompressedVideo;

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

class Subscriber : public image_transport::SimpleSubscriberPlugin<CompressedVideo>
{
public:
  Subscriber();
  ~Subscriber() override;

  std::string getTransportName() const override { return "nova"; }

protected:
  void internalCallback(
    const CompressedVideo::ConstSharedPtr & msg, const Callback & user_callback) override;

  void subscribeImpl(
    NodeType node, const std::string & base_topic, const Callback & callback, QoSType custom_qos,
    rclcpp::SubscriptionOptions) override;

  void shutdown() override;

private:
  void frame_ready(const Image::ConstSharedPtr & img, bool /*is_key_frame */) const;

  void initialize(NodeType node, const std::string & base_topic);

  std::string get_decoders_from_map(const std::string & encoding);

  NodeType node_;
  ffmpeg_encoder_decoder::Decoder decoder_;
  std::string decoder_type_;
  const Callback * user_callback_;
  uint64_t pts_{0};
  std::string parameter_namespace_;
};
}  // namespace nova::ros
#endif  // NOVA_ROS__SUBSCRIBER_HPP_
