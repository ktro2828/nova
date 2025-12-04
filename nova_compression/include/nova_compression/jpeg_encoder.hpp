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

#ifndef NOVA_COMPRESSION__JPEG_ENCODER_HPP_
#define NOVA_COMPRESSION__JPEG_ENCODER_HPP_

#include <nova_common/datatype.hpp>
#include <nova_common/format.hpp>

#include <string>
#include <utility>

#if defined(JETSON_AVAILABLE) || defined(NVJPEG_AVAILABLE)
#include <cuda_runtime.h>
#endif

#ifdef TURBOJPEG_AVAILABLE
#include <turbojpeg.h>
#endif

#ifdef JETSON_AVAILABLE
#include <cuda/api.hpp>

#include <NvJpegEncoder.h>
#include <nppi_support_functions.h>
#endif

#ifdef NVJPEG_AVAILABLE
#include <nvjpeg.h>
#endif

namespace nova::compression
{
/**
 * @brief Base class for JPEG encoders.
 */
class JpegEncoderBase
{
public:
  explicit JpegEncoderBase(std::string name) : name_(std::move(name)) {}
  virtual CompressedImage::UniquePtr encode(const Image & img, int quality, ImageFormat format) = 0;

protected:
  std::string name_;
};

#ifdef TURBOJPEG_AVAILABLE
/**
 * @brief CPU-based JPEG encoder using TurboJPEG library.
 */
class CpuJpegEncoder final : public JpegEncoderBase
{
public:
  CpuJpegEncoder(std::string name);
  ~CpuJpegEncoder();

  CompressedImage::UniquePtr encode(
    const Image & msg, int quality = 90, ImageFormat format = ImageFormat::RGB) override;

private:
  tjhandle handle_;         //!< TurboJPEG handle.
  unsigned char * buffer_;  //!< Buffer for storing encoded JPEG data.
  unsigned long size_;      //!< Size of the encoded JPEG data.
};
#endif  // TURBOJPEG_AVAILABLE

#ifdef JETSON_AVAILABLE
/**
 * @brief JPEG encoder using Jetson Inference library.
 */
class JetsonJpegEncoder final : public JpegEncoderBase
{
public:
  explicit JetsonJpegEncoder(std::string name);
  ~JetsonJpegEncoder();

  CompressedImage::UniquePtr encode(
    const Image & msg, int quality = 90, ImageFormat format = ImageFormat::RGB) override;

private:
  NvJPEGEncoder * encoder_;            //!< NvJPEG encoder handle.
  size_t image_size_;                  //!< Size of the input image data.
  size_t yuv_size_;                    //!< Size of the YUV data.
  Npp8u * image_d_;                    //!< Input image data in device memory.
  std::array<Npp8u, 3> yuv_d_;         //!< YUV data in device memory.
  std::array<void *, 3> yuv_h_;        //!< YUV data in host memory.
  int image_step_bytes_;               //!< Step size in bytes for the input image data.
  std::array<int, 3> yuv_step_bytes_;  //!< Step sizes in bytes for the YUV data.

  cudaStream_t stream_;             //!< CUDA stream for asynchronous operations.
  NppStreamContext context_;        //!< NPP stream context for asynchronous operations.
  std::optional<NvBuffer> buffer_;  //!< Optional NvBuffer for storing encoded JPEG data.
};
#endif  // JETSON_AVAILABLE

#ifdef NVJPEG_AVAILABLE
/**
 * JPEG encoder using NVIDIA NVJPEG library.
 */
class NvJpegEncoder final : public JpegEncoderBase
{
public:
  NvJpegEncoder(std::string name);
  ~NvJpegEncoder();

  CompressedImage::UniquePtr encode(
    const Image & msg, int quality = 90, ImageFormat format = ImageFormat::RGB) override;

private:
  // void setNVJPEGParams(int quality, ImageFormat format);
  void set_nv_image(const Image & msg);

  cudaStream_t stream_;
  nvjpegHandle_t handle_;
  nvjpegEncoderState_t state_;
  nvjpegEncoderParams_t params_;
  nvjpegInputFormat_t input_format_;
  nvjpegChromaSubsampling_t subsampling_;
  nvjpegImage_t nv_image_;
};
#endif  // NVJPEG_AVAILABLE
}  // namespace nova::compression
#endif  // NOVA_COMPRESSION__JPEG_ENCODER_HPP_
