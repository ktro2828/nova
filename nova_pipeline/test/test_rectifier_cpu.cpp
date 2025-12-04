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

#include "nova_pipeline/rectifier.hpp"

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace nova::pipeline
{
static sensor_msgs::msg::CameraInfo make_identity_camera_info(
  std::uint32_t width, std::uint32_t height)
{
  sensor_msgs::msg::CameraInfo info;
  info.width = width;
  info.height = height;

  // Identity intrinsic matrix K
  info.k = {1.0, 0.0, static_cast<double>(width) / 2.0,
            0.0, 1.0, static_cast<double>(height) / 2.0,
            0.0, 0.0, 1.0};

  // Rectification matrix R as identity
  info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // Projection matrix P (identity-like with principal point and focal length 1)
  info.p = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

  // Zero distortion coefficients (no distortion)
  info.distortion_model = "plumb_bob";
  info.d = {0.0, 0.0, 0.0, 0.0, 0.0};

  return info;
}

static sensor_msgs::msg::Image make_rgb8_image(std::uint32_t width, std::uint32_t height)
{
  sensor_msgs::msg::Image img;
  img.width = width;
  img.height = height;
  img.encoding = "rgb8";
  img.is_bigendian = 0;
  img.step = width * 3;
  img.data.resize(static_cast<std::size_t>(img.step * img.height));

  // Fill with deterministic gradient
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y * img.step + x * 3);
      img.data[idx + 0] =
        static_cast<std::uint8_t>((x * 255) / (width > 1 ? (width - 1) : 1));  // R
      img.data[idx + 1] =
        static_cast<std::uint8_t>((y * 255) / (height > 1 ? (height - 1) : 1));  // G
      img.data[idx + 2] = 127;                                                   // B
    }
  }
  return img;
}

TEST(OpenCVRectifierCPUTest, CanConstructWithIdentityCameraInfo)
{
  const auto cam = make_identity_camera_info(16, 12);
  OpenCVRectifierCPU rectifier(cam, /*alpha=*/0.0);

  EXPECT_TRUE(rectifier.is_camera_info_ready());
  const auto & ci = rectifier.camera_info();
  EXPECT_EQ(ci.width, cam.width);
  EXPECT_EQ(ci.height, cam.height);
  EXPECT_EQ(ci.distortion_model, "plumb_bob");
}

TEST(OpenCVRectifierCPUTest, RectifyRgb8ImageBasic)
{
  const std::uint32_t width = 16;
  const std::uint32_t height = 12;
  const auto cam = make_identity_camera_info(width, height);
  OpenCVRectifierCPU rectifier(cam, /*alpha=*/0.0);

  auto input = make_rgb8_image(width, height);

  auto output = rectifier.rectify(input);
  ASSERT_NE(output, nullptr) << "Rectifier returned null image";
  EXPECT_EQ(output->width, input.width);
  EXPECT_EQ(output->height, input.height);
  EXPECT_EQ(output->encoding, input.encoding);
  EXPECT_EQ(output->is_bigendian, input.is_bigendian);
  EXPECT_EQ(output->step, input.step);
  EXPECT_EQ(output->data.size(), input.data.size());
}

TEST(OpenCVRectifierCPUTest, RectifyPropagatesHeader)
{
  const std::uint32_t width = 8;
  const std::uint32_t height = 8;
  const auto cam = make_identity_camera_info(width, height);
  OpenCVRectifierCPU rectifier(cam, /*alpha=*/0.0);

  auto input = make_rgb8_image(width, height);
  input.header.frame_id = "camera_frame";
  // Note: stamp is zero by default; we just check propagation
  auto output = rectifier.rectify(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->header.frame_id, input.header.frame_id);
  EXPECT_EQ(output->header.stamp, input.header.stamp);
}
}  // namespace nova::pipeline
