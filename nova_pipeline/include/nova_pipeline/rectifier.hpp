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
/**
 * @brief Base class for rectifiers.
 */
class RectifierBase
{
public:
  /**
   * @brief Destructor.
   */
  virtual ~RectifierBase() {}

  /**
   * @brief Get the backend name.
   *
   * @return The backend name.
   */
  virtual const char * backend() const noexcept = 0;

  /**
   * @brief Rectify the image.
   *
   * @param msg The input image.
   * @return The rectified image.
   */
  virtual Image::UniquePtr rectify(const Image & msg) = 0;

  /**
   * @brief Check if the camera information is ready.
   *
   * @return True if camera information is ready, false otherwise.
   */
  bool is_camera_info_ready() const noexcept { return camera_info_rect_.has_value(); }

  /**
   * @brief Get the camera information.
   *
   * @return The camera information.
   */
  CameraInfo camera_info() { return camera_info_rect_.value(); }

  /**
   * @brief Get the read-only reference to camera information.
   *
   * @return The read-only reference to camera information.
   */
  const CameraInfo & camera_info() const { return camera_info_rect_.value(); }

protected:
  std::optional<CameraInfo> camera_info_rect_{
    std::nullopt};  //!< Camera information for rectification.
};

/**
 * @brief OpenCV rectifier for CPU.
 */
class OpenCVRectifierCPU : public RectifierBase
{
public:
  /**
   * @brief Constructor.
   *
   * @param info Camera information for rectification.
   * @param alpha Rectification alpha parameter.
   */
  explicit OpenCVRectifierCPU(const CameraInfo & info, double alpha = 0.0);

  /**
   * @brief Get the backend name.
   *
   * @return The backend name.
   */
  const char * backend() const noexcept override;

  /**
   * @brief Rectify the input image.
   *
   * @param msg The input image to be rectified.
   * @return The rectified image.
   */
  Image::UniquePtr rectify(const Image & msg) override;

private:
  cv::Mat map_x_;              //!< Map for x-coordinate rectification.
  cv::Mat map_y_;              //!< Map for y-coordinate rectification.
  cv::Mat camera_intrinsics_;  //!< Camera intrinsics matrix.
  cv::Mat distortion_coeffs_;  //!< Distortion coefficients matrix.
};

#ifdef OPENCV_CUDA_AVAILABLE
/**
 * @brief OpenCV-based rectifier using CUDA.
 */
class OpenCVRectifierGPU : public RectifierBase
{
public:
  /**
   * @brief Constructor.
   *
   * @param info Camera information for rectification.
   * @param alpha Rectification alpha parameter.
   */
  explicit OpenCVRectifierGPU(const CameraInfo & info, double alpha = 0.0);

  /**
   * @brief Get the backend name.
   *
   * @return The backend name.
   */
  const char * backend() const noexcept override;

  /**
   * @brief Rectify the input image using CUDA.
   *
   * @param msg The input image to be rectified.
   * @return The rectified image.
   */
  Image::UniquePtr rectify(const Image & msg) override;

private:
  cv::cuda::GpuMat map_x_;  //!< Map for x-coordinate rectification.
  cv::cuda::GpuMat map_y_;  //!< Map for y-coordinate rectification.
};
#endif  // OPENCV_CUDA_AVAILABLE

#if NPP_AVAILABLE
/**
 * @brief NVIDIA NPP-based rectifier.
 */
class NPPRectifier : public RectifierBase
{
public:
  /**
   * @brief Constructor for NPPRectifier.
   *
   * @param width The width of the input image.
   * @param height The height of the input image.
   * @param map_x The x-coordinate map.
   * @param map_y The y-coordinate map.
   */
  NPPRectifier(int width, int height, const Npp32f * map_x, const Npp32f * map_y);

  /**
   * @brief Constructor for NPPRectifier.
   *
   * @param info The camera information.
   * @param alpha The alpha value.
   */
  explicit NPPRectifier(const CameraInfo & info, double alpha = 0.0);

  /**
   * @brief Destructor for NPPRectifier.
   */
  ~NPPRectifier();

  /**
   * @brief Get the backend name.
   *
   * @return The backend name.
   */
  const char * backend() const noexcept override;

  /**
   * @brief Rectify the input image.
   *
   * @param msg The input image.
   * @return The rectified image.
   */
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
#endif  // NPP_AVAILABLE
}  // namespace nova::pipeline
#endif  // NOVA_PIPELINE__RECTIFIER_HPP_
