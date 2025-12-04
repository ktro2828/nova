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

#ifndef NOVA_ROS__IMGPROC_NODE_HPP_
#define NOVA_ROS__IMGPROC_NODE_HPP_

#include "nova_ros/task_queue.hpp"

#include <nova_common/datatype.hpp>
#include <nova_compression/builder.hpp>
#include <nova_compression/jpeg_encoder.hpp>
#include <nova_pipeline/builder.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>
#include <optional>
#include <thread>

namespace nova::ros
{
class ImgProcNode : public rclcpp::Node
{
public:
  explicit ImgProcNode(const rclcpp::NodeOptions & options);
  ~ImgProcNode();

private:
  void determine_qos();
  void on_image(Image::ConstSharedPtr msg, int32_t quality);
  void on_camera_info(CameraInfo::ConstSharedPtr msg, double alpha);

  // --- Rectifier ---
  std::shared_ptr<pipeline::RectifierBase> rectifier_;

  // --- JPEG Encoder ---
  std::shared_ptr<compression::JpegEncoderBase> raw_jpeg_encoder_;
  std::shared_ptr<compression::JpegEncoderBase> rectified_jpeg_encoder_;

  // --- Subscriptions ---
  rclcpp::Subscription<Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_subscription_;
  // --- Publishers ---
  rclcpp::Publisher<Image>::SharedPtr rectified_image_publisher_;
  rclcpp::Publisher<CompressedImage>::SharedPtr raw_compressed_image_publisher_;
  rclcpp::Publisher<CompressedImage>::SharedPtr rectified_compressed_image_publisher_;
  rclcpp::Publisher<CameraInfo>::SharedPtr camera_info_publisher_;

  rclcpp::TimerBase::SharedPtr qos_request_timer_;  //!< Timer for requesting QoS

  // --- Task Queues & Workers ---
  std::optional<TaskQueue> compress_task_queue_;
  std::optional<std::thread> compress_worker_;
  std::optional<TaskQueue> rectify_task_queue_;
  std::optional<std::thread> rectify_worker_;
};
}  // namespace nova::ros
#endif  // NOVA_ROS__IMGPROC_NODE_HPP_
