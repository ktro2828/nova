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

#include "nova_ros/subscriber.hpp"

#include "nova_ros/parameter_definition.hpp"

#include <ffmpeg_encoder_decoder/utils.hpp>

#include <functional>
#include <stdexcept>
#include <string>

namespace nova::ros
{
using std::placeholders::_1;
using std::placeholders::_2;

using ParameterValue = ParameterDefinition::ParameterValue;
using ParameterDescriptor = ParameterDefinition::ParameterDescriptor;

Subscriber::Subscriber()
{
}

Subscriber::~Subscriber()
{
  decoder_.reset();
}

void Subscriber::shutdown()
{
  if (decoder_.isInitialized()) {
    decoder_.flush();  // may cause additional frame_ready() calls!
    decoder_.reset();
  }
  SimpleSubscriberPlugin::shutdown();
}

void Subscriber::frame_ready(const Image::ConstSharedPtr & img, bool) const
{
  (*user_callback_)(img);
}

void Subscriber::subscribeImpl(
  NodeType node, const std::string & base_topic, const Callback & callback, QoSType custom_qos,
  rclcpp::SubscriptionOptions opt)
{
  initialize(node, base_topic);
#ifdef IMAGE_TRANSPORT_NEEDS_PUBLISHEROPTIONS
  image_transport::SimpleSubscriberPlugin<CompressedVideo>::subscribeImpl(
    node, base_topic, callback, custom_qos, opt);
#else
  (void)opt;  // to suppress compiler warning
  image_transport::SimpleSubscriberPlugin<CompressedVideo>::subscribeImpl(
    node, base_topic, callback, custom_qos);
#endif
}

void Subscriber::initialize(NodeType node, const std::string & base_topic_o)
{
  node_ = node;
#ifdef IMAGE_TRANSPORT_RESOLVES_BASE_TOPIC
  const std::string base_topic = base_topic_o;
#else
  const std::string base_topic =
    node_->get_node_topics_interface()->resolve_topic_name(base_topic_o);
#endif
#ifdef IMAGE_TRANSPORT_USE_NODEINTERFACE
  uint ns_len = std::string(node_.get_node_base_interface()->get_namespace()).length();
#else
  uint ns_len = node_->get_effective_namespace().length();
#endif
  // if a namespace is given (ns_len > 1), then strip one more
  // character to avoid a leading "/" that will then become a "."
  uint ns_prefix_len = ns_len > 1 ? ns_len + 1 : ns_len;
  std::string param_base_name = base_topic.substr(ns_prefix_len);
  std::replace(param_base_name.begin(), param_base_name.end(), '/', '.');
  parameter_namespace_ = param_base_name + "." + getTransportName() + ".";
}

std::string Subscriber::get_decoders_from_map(const std::string & encoding)
{
  const auto splits = ffmpeg_encoder_decoder::utils::split_encoding(encoding);
  std::string decoders;
  for (size_t i = splits.size(); i > 0; --i) {
    std::string p_name;
    for (size_t j = 0; j < i; ++j) {
      p_name += (j == 0 ? "." : "_") + splits[j];
    }
    ParameterDefinition param{
      ParameterValue(""),
      ParameterDescriptor()
        .set__name("decoders" + p_name)
        .set__type(rcl_interfaces::msg::ParameterType::PARAMETER_STRING)
        .set__description("decoders for encoding: " + p_name)
        .set__read_only(false),
      ""};
    decoders = param.declare_parameter(node_, parameter_namespace_).get<std::string>();
    if (!decoders.empty()) {
      break;
    }
  }
  return decoders;
}

void Subscriber::internalCallback(
  const CompressedVideo::ConstSharedPtr & msg, const Callback & user_callback)
{
  if (decoder_.isInitialized()) {
    // the decoder is already initialized
    decoder_.decodePacket(
      msg->format, &msg->data[0], msg->data.size(), pts_++, msg->frame_id,
      rclcpp::Time(msg->timestamp));
    return;
  }
  user_callback_ = &user_callback;
  const auto codec = ffmpeg_encoder_decoder::utils::split_encoding(msg->format)[0];
  auto decoder_names = this->get_decoders_from_map(msg->format);
  decoder_names = ffmpeg_encoder_decoder::utils::filter_decoders(codec, decoder_names);
  if (decoder_names.empty()) {
    decoder_names = ffmpeg_encoder_decoder::utils::find_decoders(codec);
    // no decoders configured for encoding
  }
  if (decoder_names.empty()) {
    // cannot find valid decoder for codec
    return;
  }

  for (const auto & dec : ffmpeg_encoder_decoder::utils::split_decoders(decoder_names)) {
    try {
      if (!decoder_.initialize(
            msg->format, std::bind(&Subscriber::frame_ready, this, _1, _2), dec)) {
        // cannot initialize decoder
        continue;
      }
    } catch (std::runtime_error &) {
      continue;
    }
    // sometimes the failure is only detected when the decoding is happening.
    // hopefully this is on the first packet.
    if (!decoder_.decodePacket(
          msg->format, &msg->data[0], msg->data.size(), pts_++, msg->frame_id,
          rclcpp::Time(msg->timestamp))) {
      decoder_.reset();
      continue;
    }
    break;
  }
}
}  // namespace nova::ros
