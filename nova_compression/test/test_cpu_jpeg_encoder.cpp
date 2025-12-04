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

#include "nova_compression/jpeg_encoder.hpp"

#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <gtest/gtest.h>

#ifdef TURBOJPEG_AVAILABLE
namespace nova::compression
{
// Helper to construct a simple RGB image with deterministic data
static sensor_msgs::msg::Image make_test_image_rgb8(std::uint32_t width, std::uint32_t height)
{
  sensor_msgs::msg::Image img;
  img.width = width;
  img.height = height;
  img.encoding = "rgb8";
  img.is_bigendian = 0;
  img.step = width * 3;  // 3 channels (RGB), 1 byte per channel
  img.data.resize(static_cast<size_t>(img.step * img.height));

  // Fill with a gradient pattern to avoid all-zero data
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y * img.step + x * 3);
      img.data[idx + 0] = static_cast<std::uint8_t>((x * 255) / (width ? width - 1 : 1));    // R
      img.data[idx + 1] = static_cast<std::uint8_t>((y * 255) / (height ? height - 1 : 1));  // G
      img.data[idx + 2] = static_cast<std::uint8_t>(127);                                    // B
    }
  }
  return img;
}

TEST(CpuJpegEncoderTest, EncodeBasicRgbImage)
{
  CpuJpegEncoder encoder;

  const auto img = make_test_image_rgb8(16, 16);
  auto compressed = encoder.encode(
    img,
    /*quality=*/90,
    /*format=*/TJPF_RGB,
    /*sampling=*/TJ_420);

  ASSERT_NE(compressed, nullptr) << "Encoder returned null CompressedImage";
  EXPECT_EQ(compressed->format, "jpeg");

  // Expect some non-trivial output size for a 16x16 RGB image at quality 90 and 4:2:0 sampling.
  EXPECT_GT(compressed->data.size(), 0U) << "Encoded JPEG output should not be empty";

  // Check header propagation
  EXPECT_EQ(compressed->header.stamp, img.header.stamp);
  EXPECT_EQ(compressed->header.frame_id, img.header.frame_id);
}

TEST(CpuJpegEncoderTest, EncodeDifferentQualityAndSampling)
{
  CpuJpegEncoder encoder;

  const auto img = make_test_image_rgb8(32, 24);

  // Encode with quality 50 and 4:4:4 sampling for variety
  auto compressed = encoder.encode(
    img,
    /*quality=*/50,
    /*format=*/TJPF_RGB,
    /*sampling=*/TJ_444);

  ASSERT_NE(compressed, nullptr);
  EXPECT_EQ(compressed->format, "jpeg");
  EXPECT_GT(compressed->data.size(), 0U);

  // Optionally compare sizes between different settings
  auto compressed_90_420 = encoder.encode(img, 90, TJPF_RGB, TJ_420);
  ASSERT_NE(compressed_90_420, nullptr);
  EXPECT_GT(compressed_90_420->data.size(), 0U);

  // Not asserting strict size ordering (depends on content), but ensure both produced data
  EXPECT_NE(compressed->data.size(), compressed_90_420->data.size());
}
}  // namespace nova::compression
#else   // TURBOJPEG_AVAILABLE
// If TurboJPEG is not available, skip tests to avoid failures in environments without the
// dependency.
TEST(CpuJpegEncoderDisabledTest, TurboJpegUnavailable)
{
  GTEST_SKIP() << "TurboJPEG not available. Skipping CpuJpegEncoder tests.";
}
#endif  // TURBOJPEG_AVAILABLE
