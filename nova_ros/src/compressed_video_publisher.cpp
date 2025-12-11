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

#include "nova_ros/compressed_video_publisher.hpp"

#include "nova_ros/parameter_definition.hpp"
#include "nova_ros/utility.hpp"

#include <algorithm>
#include <ctime>
#include <functional>
#include <memory>
#include <string>

namespace nova::ros
{
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;
using std::placeholders::_5;
using std::placeholders::_6;
using std::placeholders::_7;
using std::placeholders::_8;
using std::placeholders::_9;

using ParameterValue = ParameterDefinition::ParameterValue;
using ParameterDescriptor = ParameterDefinition::ParameterDescriptor;
using ParameterType = rcl_interfaces::msg::ParameterType;

static const ParameterDefinition params[] = {
  {ParameterValue("libx264"),
   ParameterDescriptor()
     .set__name("encoder")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("ffmpeg encoder to use, see supported encoders")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("encoder_av_options")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("comma-separated list of AV options: profile:main,preset:ll")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("preset")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("ffmpeg encoder preset")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("tune")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("ffmpeg encoder tune")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("delay")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("ffmpeg encoder delay")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("crf")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("ffmpeg encoder crf")
     .set__read_only(false),
   ""},
  {ParameterValue(""),
   ParameterDescriptor()
     .set__name("pixel_format")
     .set__type(ParameterType::PARAMETER_STRING)
     .set__description("pixel format to use for encoding")
     .set__read_only(false),
   ""},
  {ParameterValue(static_cast<int>(10)),
   ParameterDescriptor()
     .set__name("qmax")
     .set__type(ParameterType::PARAMETER_INTEGER)
     .set__description("max video quantizer scale, see ffmpeg docs")
     .set__read_only(false)
     .set__integer_range(
       {rcl_interfaces::msg::IntegerRange().set__from_value(-1).set__to_value(1024).set__step(1)}),
   ""},
  {ParameterValue(static_cast<int64_t>(8242880)),
   ParameterDescriptor()
     .set__name("bit_rate")
     .set__type(ParameterType::PARAMETER_INTEGER)
     .set__description("target bit rate, see ffmpeg docs")
     .set__read_only(false)
     .set__integer_range({rcl_interfaces::msg::IntegerRange()
                            .set__from_value(1)
                            .set__to_value(std::numeric_limits<int>::max())
                            .set__step(1)}),
   ""},
  {ParameterValue(static_cast<int>(1)),
   ParameterDescriptor()
     .set__name("gop_size")
     .set__type(ParameterType::PARAMETER_INTEGER)
     .set__description("gop size (distance between keyframes)")
     .set__read_only(false)
     .set__integer_range({rcl_interfaces::msg::IntegerRange()
                            .set__from_value(1)
                            .set__to_value(std::numeric_limits<int>::max())
                            .set__step(1)}),
   "gop_size not set, defaulting to 1!"},
};

CompressedVideoPublisher::CompressedVideoPublisher()
{
}

CompressedVideoPublisher::~CompressedVideoPublisher()
{
}

void CompressedVideoPublisher::declare_parameter(
  NodeType node, const ParameterDefinition & definition)
{
  const auto value = definition.declare_parameter(node, parameter_namespace_);
  const auto & name = definition.descriptor.name;
  if (name == "encoding" || name == "encoder") {
    encoder_.setEncoder(value.get<std::string>());
  } else if (name == "encoder_av_options") {
    handle_av_options(value.get<std::string>());
  } else if (name == "preset" || name == "tune" || name == "delay" || name == "crf") {
    if (!value.get<std::string>().empty()) {
      encoder_.addAVOption(name, value.get<std::string>());
    }
  } else if (name == "pixel_format") {
    encoder_.setAVSourcePixelFormat(value.get<std::string>());
  } else if (name == "qmax") {
    encoder_.setQMax(value.get<int>());
  } else if (name == "gop_size") {
    encoder_.setGOPSize(value.get<int>());
  } else {
    RCLCPP_ERROR_STREAM(node->get_logger(), "unknown parameter: " << name);
  }
}

void CompressedVideoPublisher::handle_av_options(const std::string & options)
{
  const auto split = utility::split_av_options(options);
  for (const auto & option : split) {
    const auto parts = utility::split_av_option(option);
    if (parts.size() == 2) {
      encoder_.addAVOption(parts[0], parts[1]);
    }
  }
}

void CompressedVideoPublisher::callback(
  const std::string & frame_id, const rclcpp::Time & stamp, const std::string & /*codec*/,
  uint32_t /*width*/, uint32_t /*height*/, uint64_t /*pts*/, uint8_t /*flags*/, uint8_t * data,
  size_t sz)
{
  auto msg = std::make_shared<CompressedVideo>();
  msg->frame_id = frame_id;
  msg->timestamp = stamp;
  msg->format = "h264";
  msg->data.assign(data, data + sz);
#ifdef IMAGE_TRANSPORT_USE_PUBLISHER_T
  (*publish_function_)->publish(*msg);
#else
  (*publish_function_)(*msg);
#endif
}

#ifdef IMAGE_TRANSPORT_NEEDS_PUBLISHEROPTIONS
void CompressedVideoPublisher::advertiseImpl(
  NodeType node, const std::string & base_topic, QoSType custom_qos, rclcpp::PublisherOptions opt)
{
  auto qos = initialize(node, base_topic, custom_qos);
  SimplePublisherPlugin<CompressedVideo>::advertiseImpl(node, base_topic, qos, opt);
}
#else
void CompressedVideoPublisher::advertiseImpl(
  NodeType node, const std::string & base_topic, rmw_qos_profile_t custom_qos)
{
  auto qos = initialize(node, base_topic, custom_qos);
  SimplePublisherPlugin<CompressedVideo>::advertiseImpl(node, base_topic, qos);
}
#endif

CompressedVideoPublisher::QoSType CompressedVideoPublisher::initialize(
  NodeType node, const std::string & base_topic, QoSType custom_qos)
{
  // namespace handling code lifted from compressed_image_transport
#ifdef IMAGE_TRANSPORT_USE_NODEINTERFACE
  uint ns_len = std::string(node.get_node_base_interface()->get_namespace()).length();
#else
  uint ns_len = node->get_effective_namespace().length();
#endif
  // if a namespace is given (ns_len > 1), then strip one more
  // character to avoid a leading "/" that will then become a "."
  uint ns_prefix_len = ns_len > 1 ? ns_len + 1 : ns_len;
  std::string param_base_name = base_topic.substr(ns_prefix_len);
  std::replace(param_base_name.begin(), param_base_name.end(), '/', '.');
  parameter_namespace_ = param_base_name + "." + getTransportName() + ".";

  for (const auto & p : params) {
    declare_parameter(node, p);
  }
  // bump queue size to 2 * distance between keyframes
#ifdef IMAGE_TRANSPORT_USE_QOS
  custom_qos.keep_last(
    std::max(static_cast<int>(custom_qos.get_rmw_qos_profile().depth), 2 * encoder_.getGOPSize()));
#else
  custom_qos.depth = std::max(static_cast<int>(custom_qos.depth), 2 * encoder_.getGOPSize());
#endif
  return (custom_qos);
}

void CompressedVideoPublisher::publish(const Image & msg, const PublisherTFn & publish_fn) const
{
  CompressedVideoPublisher * me = const_cast<CompressedVideoPublisher *>(this);
  me->publish_function_ = &publish_fn;
  if (!me->encoder_.isInitialized()) {
    if (!me->encoder_.initialize(
          msg.width, msg.height,
          std::bind(&CompressedVideoPublisher::callback, me, _1, _2, _3, _4, _5, _6, _7, _8, _9),
          msg.encoding)) {
      return;
    }
  }
  // may trigger packetReady() callback(s) from encoder!
  me->encoder_.encodeImage(msg);
}
}  // namespace nova::ros

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(nova::ros::CompressedVideoPublisher, image_transport::PublisherPlugin)
