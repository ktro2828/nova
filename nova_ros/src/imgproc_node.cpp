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

#include "nova_ros/imgproc_node.hpp"

#include "nova_ros/qos.hpp"
#include "nova_ros/task_queue.hpp"

#include <nova_common/datatype.hpp>
#include <nova_common/format.hpp>

#include <algorithm>
#include <utility>

namespace nova::ros
{
ImgProcNode::ImgProcNode(const rclcpp::NodeOptions & options) : Node("imgproc_node", options)
{
  raw_jpeg_encoder_ = compression::build_jpeg_encoder("raw_jpeg_encoder");
  rectified_jpeg_encoder_ = compression::build_jpeg_encoder("rectified_jpeg_encoder");

  // QoS
  qos_request_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::milliseconds(100), [this]() { this->determine_qos(); });
}

ImgProcNode::~ImgProcNode()
{
  if (compress_task_queue_) {
    compress_task_queue_->stop();
  }
  if (compress_worker_) {
    compress_worker_->join();
  }
  if (rectify_task_queue_) {
    rectify_task_queue_->stop();
  }
  if (rectify_worker_) {
    rectify_worker_->join();
  }
}

void ImgProcNode::determine_qos()
{
  auto image_topic = this->get_node_topics_interface()->resolve_topic_name("image_raw", false);
  auto info_topic = this->get_node_topics_interface()->resolve_topic_name("camera_info", false);

  auto image_qos_opt = find_qos(this, image_topic);
  auto info_qos_opt = find_qos(this, info_topic);
  if (!image_qos_opt || !info_qos_opt) {
    return;
  } else {
    // parameters
    size_t max_task_queue_size = this->declare_parameter<int64_t>("max_task_queue_size", 5);
    int32_t quality = this->declare_parameter<int32_t>("jpeg_encoder.quality", 90);
    double alpha = this->declare_parameter<double>("rectifier.alpha", 0.0);

    // initialize task queues & workers
    compress_task_queue_.emplace(max_task_queue_size);
    compress_worker_.emplace(&TaskQueue::run, &compress_task_queue_.value());
    rectify_task_queue_.emplace(max_task_queue_size);
    rectify_worker_.emplace(&TaskQueue::run, &rectify_task_queue_.value());

    // initialize subscriptions & publishers
    const auto image_qos = image_qos_opt.value();
    const auto info_qos = info_qos_opt.value();

    image_subscription_ = this->create_subscription<Image>(
      image_topic, image_qos,
      [this, quality](Image::ConstSharedPtr msg) { this->on_image(msg, quality); });
    camera_info_subscription_ = this->create_subscription<CameraInfo>(
      info_topic, info_qos,
      [this, alpha](CameraInfo::ConstSharedPtr msg) { this->on_camera_info(msg, alpha); });

    rectified_image_publisher_ = this->create_publisher<Image>("image_rect", image_qos);
    raw_compressed_image_publisher_ =
      this->create_publisher<CompressedImage>("image_raw/compressed", image_qos);
    rectified_compressed_image_publisher_ =
      this->create_publisher<CompressedImage>("image_rect/compressed", image_qos);

    // once all queries received, stop the timer
    qos_request_timer_->cancel();
  }
}

void ImgProcNode::on_image(sensor_msgs::msg::Image::ConstSharedPtr msg, int32_t quality)
{
  ImageFormat image_format;
  if (msg->encoding == "rgb8") {
    image_format = ImageFormat::RGB;
  } else if (msg->encoding == "bgr8") {
    image_format = ImageFormat::BGR;
  } else {
    RCLCPP_ERROR_STREAM(this->get_logger(), "Unsupported image encoding: " << msg->encoding);
    return;
  }

  if (compress_task_queue_) {
    compress_task_queue_->add_task([this, msg, quality, image_format]() {
      auto raw_comp_img = this->raw_jpeg_encoder_->encode(*msg, quality, image_format);
      raw_compressed_image_publisher_->publish(std::move(raw_comp_img));
    });
  }

  if (rectify_task_queue_) {
    if (!rectifier_) {
      RCLCPP_WARN(this->get_logger(), "Rectifier not initialized");
      return;
    }

    rectify_task_queue_->add_task([this, msg, quality, image_format]() {
      auto rect_img = this->rectifier_->rectify(*msg);
      auto rect_info = this->rectifier_->camera_info();
      rect_info.header = rect_img->header;

      auto rect_comp_img = this->rectified_jpeg_encoder_->encode(*rect_img, quality, image_format);

      rectified_image_publisher_->publish(std::move(rect_img));
      camera_info_publisher_->publish(std::move(rect_info));
      rectified_compressed_image_publisher_->publish(std::move(rect_comp_img));
    });
  }
}

void ImgProcNode::on_camera_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, double alpha)
{
  if (msg->d.size() == 0 || msg->p.size() == 0) {
    RCLCPP_ERROR_STREAM(
      this->get_logger(), "Camera info message does not contain distortion or projection matrix");
    return;
  }
  rectifier_ = pipeline::build_rectifier(*msg, alpha);

  // unsubscribe
  camera_info_subscription_.reset();
}
}  // namespace nova::ros

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(nova::ros::ImgProcNode)
