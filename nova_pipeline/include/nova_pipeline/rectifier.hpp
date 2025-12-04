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

#ifndef NOVA_PIPELINE__RECTIFIER_HPP_
#define NOVA_PIPELINE__RECTIFIER_HPP_

#include <nova_common/datatype.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <optional>

#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/core/cuda.hpp>
#endif

#ifdef NPP_AVAILABLE
#include <cuda_runtime.h>
#include <nppdefs.h>
#endif

namespace nova::pipeline
{
class RectifierBase
{
public:
  virtual ~RectifierBase() {}
  virtual Image::UniquePtr rectify(const Image & msg) = 0;
  bool is_camera_info_ready() const noexcept { return camera_info_rect_.has_value(); }
  CameraInfo & camera_info() { return camera_info_rect_.value(); }
  const CameraInfo & camera_info() const { return camera_info_rect_.value(); }

protected:
  std::optional<CameraInfo> camera_info_rect_{std::nullopt};
};

class OpenCVRectifierCPU : public RectifierBase
{
public:
  explicit OpenCVRectifierCPU(const CameraInfo & info, double alpha = 0.0);

  Image::UniquePtr rectify(const Image & msg) override;

private:
  cv::Mat map_x_;
  cv::Mat map_y_;
  cv::Mat camera_intrinsics_;
  cv::Mat distortion_coeffs_;
};

#ifdef OPENCV_CUDA_AVAILABLE
class OpenCVRectifierGPU : public RectifierBase
{
public:
  explicit OpenCVRectifierGPU(const CameraInfo & info, double alpha = 0.0);

  Image::UniquePtr rectify(const Image & msg) override;

private:
  cv::cuda::GpuMat map_x_;
  cv::cuda::GpuMat map_y_;
};
#endif

#if NPP_AVAILABLE
class NPPRectifier : public RectifierBase
{
public:
  NPPRectifier(int width, int height, const Npp32f * map_x, const Npp32f * map_y);
  explicit NPPRectifier(const CameraInfo & info, double alpha = 0.0);
  ~NPPRectifier();

  cudaStream_t & cuda_stream() { return stream_; }

  Image::UniquePtr rectify(const Image & msg) override;

private:
  Npp32f * pxl_map_x_;  //!< Pixel map for x-coordinate
  Npp32f * pxl_map_y_;  //!< Pixel map for y-coordinate
  int pxl_map_x_step_;  //!< Step size for x-coordinate map
  int pxl_map_y_step_;  //!< Step size for y-coordinate map
  Npp8u * src_;         //!< Source image data
  Npp8u * dst_;         //!< Destination image data
  int src_step_;        //!< Step size for source image data
  int dst_step_;        //!< Step size for destination image data

  cudaStream_t stream_;  //!< CUDA stream for asynchronous operations
};
#endif
}  // namespace nova::pipeline
#endif  // NOVA_PIPELINE__RECTIFIER_HPP_
