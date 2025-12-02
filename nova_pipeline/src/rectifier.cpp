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

#include <iostream>
#include <memory>

#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/flann.hpp>
#endif

#ifdef NPP_AVAILABLE
#include <npp.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#endif

namespace nova::pipeline
{
namespace
{
#define CHECK_NPP(status)                                                              \
  if (status != NPP_SUCCESS) {                                                         \
    std::cerr << "NPP error: " << status << " (" << __FILE__ << ":" << __LINE__ << ")" \
              << std::endl;                                                            \
  }

#define CHECK_CUDA(status)                                                                         \
  if (status != cudaSuccess) {                                                                     \
    std::cerr << "CUDA error: " << cudaGetErrorName(status) << " (" << __FILE__ << ":" << __LINE__ \
              << ")" << std::endl;                                                                 \
  }

static CameraInfo compute_maps(
  const CameraInfo & info, float * map_x, float * map_y, double alpha = 0.0)
{
  cv::Mat intrinsics(3, 3, CV_64F);
  cv::Mat distortion_coefficients(1, info.d.size(), CV_64F);

  // Initialize intrinsics matrix
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      intrinsics.at<double>(row, col) = info.k[row * 3 + col];
    }
  }

  // Initialize distortion coefficients
  for (size_t i = 0; i < info.d.size(); ++i) {
    distortion_coefficients.at<double>(0, i) = info.d[i];
  }

  cv::Mat new_intrinsics = cv::getOptimalNewCameraMatrix(
    intrinsics, distortion_coefficients, cv::Size(info.width, info.height), alpha);

  cv::Mat m1(info.height, info.width, CV_32FC1, map_x);
  cv::Mat m2(info.height, info.width, CV_32FC1, map_y);

  cv::initUndistortRectifyMap(
    intrinsics, distortion_coefficients, cv::Mat::eye(3, 3, CV_64F), new_intrinsics,
    cv::Size(info.width, info.height), CV_32FC1, m1, m2);

  // Copy the original camera info and update only D and K
  CameraInfo camera_info_rect(info);
  // After undistortion, the result will be as if it is captured with a camera using the camera with
  // new_intrinsics and zero distortion
  camera_info_rect.d.assign(info.d.size(), 0.0);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      camera_info_rect.k[row * 3 + col] = new_intrinsics.at<double>(row, col);
      camera_info_rect.p[row * 4 + col] = new_intrinsics.at<double>(row, col);
    }
  }

  return camera_info_rect;
}
}  // namespace

OpenCVRectifierCPU::OpenCVRectifierCPU(const CameraInfo & info, double alpha)
{
  map_x_ = cv::Mat(info.height, info.width, CV_32FC1);
  map_y_ = cv::Mat(info.height, info.width, CV_32FC1);

  camera_info_rect_ = compute_maps(info, map_x_.ptr<float>(), map_y_.ptr<float>(), alpha);
}

Image::UniquePtr OpenCVRectifierCPU::rectify(const Image & msg)
{
  Image::UniquePtr result = std::make_unique<Image>();
  result->header = msg.header;
  result->height = msg.height;
  result->width = msg.width;
  result->encoding = msg.encoding;
  result->is_bigendian = msg.is_bigendian;
  result->step = msg.step;
  result->data.resize(msg.data.size());

  cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
  cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());

  cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);

  return result;
}

#ifdef OPENCV_CUDA_AVAILABLE
OpenCVRectifierGPU::OpenCVRectifierGPU(const CameraInfo & info, double alpha)
{
  cv::Mat map_x(info.height, info.width, CV_32FC1);
  cv::Mat map_y(info.height, info.width, CV_32FC1);

  camera_info_rect_ = compute_maps(info, map_x.ptr<float>(), map_y.ptr<float>(), alpha);

  map_x_ = cv::cuda::GpuMat(map_x);
  map_y_ = cv::cuda::GpuMat(map_y);
}

Image::UniquePtr OpenCVRectifierGPU::rectify(const Image & msg)
{
  Image::UniquePtr result = std::make_unique<Image>();
  result->header = msg.header;
  result->height = msg.height;
  result->width = msg.width;
  result->encoding = msg.encoding;
  result->is_bigendian = msg.is_bigendian;
  result->step = msg.step;
  result->data.resize(msg.data.size());

  cv::Mat src(msg.height, msg.width, CV_8UC3, (void *)msg.data.data());
  cv::cuda::GpuMat src_d = cv::cuda::GpuMat(src);
  cv::cuda::GpuMat dst_d = cv::cuda::GpuMat(cv::Size(msg.width, msg.height), src.type());

  cv::cuda::remap(src_d, dst_d, map_x_, map_y_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

  // Copy back to host
  cv::Mat dst(msg.height, msg.width, CV_8UC3, (void *)result->data.data());
  dst_d.download(dst);

  return result;
}
#endif  // OPENCV_CUDA_AVAILABLE

#ifdef NPP_AVAILABLE
NPPRectifier::NPPRectifier(int width, int height, const Npp32f * map_x, const Npp32f * map_y)
: pxl_map_x_(nullptr), pxl_map_y_(nullptr)
{
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  nppSetStream(stream_);

  pxl_map_x_ = nppiMalloc_32f_C1(width, height, &pxl_map_x_step_);
  pxl_map_y_ = nppiMalloc_32f_C1(width, height, &pxl_map_y_step_);
  if (!pxl_map_x_ || !pxl_map_y_) {
    std::cerr << "Failed to allocate memory for rectification maps" << std::endl;
    return;
  }

  src_ = nppiMalloc_8u_C3(width, height, &src_step_);
  dst_ = nppiMalloc_8u_C3(width, height, &dst_step_);
  if (!src_ || !dst_) {
    std::cerr << "Failed to allocate memory for rectification images" << std::endl;
    return;
  }

  CHECK_CUDA(cudaMemcpy2DAsync(
    pxl_map_x_, pxl_map_x_step_, map_x, width * sizeof(float), width * sizeof(float), height,
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpy2DAsync(
    pxl_map_y_, pxl_map_y_step_, map_y, width * sizeof(float), width * sizeof(float), height,
    cudaMemcpyHostToDevice, stream_));
}

NPPRectifier::NPPRectifier(const CameraInfo & info, double alpha)
{
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  nppSetStream(stream_);

  pxl_map_x_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_x_step_);
  pxl_map_y_ = nppiMalloc_32f_C1(info.width, info.height, &pxl_map_y_step_);
  if (!pxl_map_x_ || !pxl_map_y_) {
    std::cerr << "Failed to allocate memory for rectification maps" << std::endl;
    return;
  }

  src_ = nppiMalloc_8u_C3(info.width, info.height, &src_step_);
  dst_ = nppiMalloc_8u_C3(info.width, info.height, &dst_step_);
  if (!src_ || !dst_) {
    std::cerr << "Failed to allocate memory for rectification images" << std::endl;
    return;
  }

  // Create rectification maps
  // TODO(someone): Verify rectification maps are correct
  float * map_x = new float[info.width * info.height];
  float * map_y = new float[info.width * info.height];
  camera_info_rect_ = compute_maps(info, map_x, map_y, alpha);

  CHECK_CUDA(cudaMemcpy2DAsync(
    pxl_map_x_, pxl_map_x_step_, map_x, info.width * sizeof(float), info.width * sizeof(float),
    info.height, cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpy2DAsync(
    pxl_map_y_, pxl_map_y_step_, map_y, info.width * sizeof(float), info.width * sizeof(float),
    info.height, cudaMemcpyHostToDevice, stream_));

  // Delete temporary memory
  delete[] map_x;
  delete[] map_y;
}

NPPRectifier::~NPPRectifier()
{
  if (pxl_map_x_) {
    nppiFree(pxl_map_x_);
  }
  if (pxl_map_y_) {
    nppiFree(pxl_map_y_);
  }
  if (src_) {
    nppiFree(src_);
  }
  if (dst_) {
    nppiFree(dst_);
  }

  cudaStreamDestroy(stream_);
}

NPPRectifier::rectify(const Image & msg)
{
  nppSetStream(stream_);

  Image::UniquePtr result = std::make_unique<Image>();
  result->header = msg.header;
  result->height = msg.height;
  result->width = msg.width;
  result->encoding = msg.encoding;
  result->is_bigendian = msg.is_bigendian;
  result->step = msg.step;
  result->data.resize(msg.data.size());

  NppiRect src_roi = {0, 0, static_cast<int>(msg.width), static_cast<int>(msg.height)};
  NppiSize src_size = {static_cast<int>(msg.width), static_cast<int>(msg.height)};
  NppiSize dst_roi_size = {static_cast<int>(msg.width), static_cast<int>(msg.height)};

  CHECK_CUDA(cudaMemcpy2DAsync(
    src_, src_step_, msg.data.data(), msg.step, msg.width * 3, msg.height, cudaMemcpyHostToDevice,
    stream_));

  NppiInterpolationMode interpolation = NPPI_INTER_LINEAR;

  CHECK_NPP(nppiRemap_8u_C3R(
    src_, src_size, src_step_, src_roi, pxl_map_x_, pxl_map_x_step_, pxl_map_y_, pxl_map_y_step_,
    dst_, dst_step_, dst_roi_size, interpolation));

  CHECK_CUDA(cudaMemcpy2DAsync(
    static_cast<void *>(result->data.data()), result->step, static_cast<const void *>(dst_),
    dst_step_, msg.width * 3 * sizeof(Npp8u), msg.height, cudaMemcpyDeviceToHost, stream_));

  return result;
}
#endif  // NPP_AVAILABLE
}  // namespace nova::pipeline
