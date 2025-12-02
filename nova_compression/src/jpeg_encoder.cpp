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

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#if defined(JETSON_AVAILABLE) || defined(NVJPEG_AVAILABLE)
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#endif

namespace nova::compression
{
namespace
{
#define TEST_ERROR(cond, str)     \
  if (cond) {                     \
    fprintf(stderr, "%s\n", str); \
  }

#define CHECK_NPP(status)                                                               \
  if (status != NPP_SUCCESS) {                                                          \
    std::cerr << "NPP error: " << status << " (" << __FILE__ << ", " << __LINE__ << ")" \
              << std::endl;                                                             \
  }

#define CHECK_CUDA(status)                                                              \
  if (status != cudaSuccess) {                                                          \
    std::cerr << "CUDA error: " << cudaGetErrorName(status) << " (" << __FILE__ << ", " \
              << __LINE__ << ")" << std::endl;                                          \
  }

#define CHECK_NVJPEG(call)                                                                \
  {                                                                                       \
    nvjpegStatus_t _e = (call);                                                           \
    if (_e != NVJPEG_STATUS_SUCCESS) {                                                    \
      std::cerr << "NVJPEG failure: \'#" << _e << "\' at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                                             \
      exit(1);                                                                            \
    }                                                                                     \
  }
}  // namespace

#ifdef TURBOJPEG_AVAILABLE
CpuJpegEncoder::CpuJpegEncoder() : buffer_(nullptr), size_(0)
{
  handle_ = tjInitCompress();
}

CpuJpegEncoder::~CpuJpegEncoder()
{
  if (buffer_) {
    tjFree(buffer_);
  }
  tjDestroy(handle_);
}

CompressedImage::UniquePtr CpuJpegEncoder::encode(
  const Image & msg, int quality, int format, int sampling)
{
  CompressedImage::UniquePtr output = std::make_unique<CompressedImage>();
  output->header = msg.header;
  output->format = "jpeg";

  if (buffer_) {
    tjFree(buffer_);
    buffer_ = nullptr;
  }

  int result = tjCompress2(
    handle_, msg.data.data(), msg.width, 0, msg.height, format, &buffer_, &size_, sampling, quality,
    TJFLAG_FASTDCT);

#if defined(LIBJPEG_TURBO_VERSION) && (LIBJPEG_TURBO_VERSION >= 2)
  TEST_ERROR(result != 0, tjGetErrorStr2(handle_));
#else
  TEST_ERROR(result != 0, tjGetErrorStr());
#endif

  output->data.resize(size_);
  memcpy(output->data.data(), buffer_, size_);

  return output;
}
#endif  // TURBOJPEG_AVAILABLE

#ifdef JETSON_AVAILABLE
JetsonJpegEncoder::JetsonJpegEncoder(std::string name)
: stream_(cuda::device::current().get().create_stream(cuda::stream::sync))
{
  encoder_ = NvJPEGEncoder::createJPEGEncoder(name.c_str());
}

JetsonJpegEncoder::~JetsonJpegEncoder()
{
  delete encoder_;

  nppiFree(image_d_);
  for (auto & p : yuv_d_) {
    nppiFree(p);
  }
}

CompressedImage::UniquePtr JetsonJpegEncoder::encode(const Image & msg, int quality, int format)
{
  CompressedImage::UniquePtr output = std::make_unique<CompressedImage>();
  output->header = msg.header;
  output->format = "jpeg";

  int width = msg.width;
  int height = msg.height;
  const auto & img = msg.data;

  if (image_size_ < img.size()) {
    // Allocate Npp8u buffers
    image_d_ = nppiMalloc_8u_c3(width, height, &image_step_bytes_);
    image_size_ = img.size();

    yuv_d_[0] = nppiMalloc_8u_C1(width, height, &yuv_step_bytes_[0]);          // Y
    yuv_d_[1] = nppiMalloc_8u_C1(width / 2, height / 2, &yuv_step_bytes_[1]);  // U
    yuv_d_[2] = nppiMalloc_8u_C1(width / 2, height / 2, &yuv_step_bytes_[2]);  // V

    // Fill element of nppStreamContext
    {
      context_.hStream = stream_.handle();
      cudaGetDevice(&context_.nCudaDeviceId);

      cudaDeviceProp prop;
      CHECK_NPP(cudaGetDeviceProperties(&prop, context_.nCudaDeviceId));
      context_.nMultiProcessorCount = prop.multiProcessorCount;
      context_.nMaxThreadsMultiProcessor = prop.maxThreadsPerMultiProcessor;
      context_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
      context_.nSharedMemPerBlock = prop.sharedMemPerBlock;

      cudaDeviceGetAttribute(
        &context_.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
        context_.nCudaDeviceId);
      cudaDeviceGetAttribute(
        &context_.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
        context_.nCudaDeviceId);
      cudaStreamGetFlags(context_.hStream, &context_.nStreamFlags);
    }

    buffer_.emplace(V4L2_PIX_FMT_YUV420M, width, height, 0);
    TEST_ERROR(buffer_->allocateMemory() != 0, "NvBuffer allocation failed");

    encoder_->setCropRect(0, 0, width, height);
  }

  CHECK_CUDA(cudaMemcpy2DAsync(
    static_cast<void *>(image_d_), image_step_bytes_, static_cast<const void *>(img.data()),
    msg.step, msg.step * sizeof(Npp8u), msg.height, cudaMemcpyHostToDevice, stream_.handle()));

  NppiSize roi = {static_cast<int>(msg.width), static_cast<int>(msg.height)};
  if (format == ImageFormat::RGB) {
    // Inplace conversion from BGR to RGB
    const int order[3] = {2, 1, 0};
    CHECK_NPP(nppiSwapChannels_8u_C3IR_Ctx(image_d_, image_step_bytes_, roi, order, context_));
  }

  // Convert RGB8 to YUV420
  CHECK_NPP(nppiRGBToYUV420_8u_C3P3R_Ctx(
    image_d_, image_step_bytes_, yuv_d_.data(), yuv_step_bytes_.data(), roi, context_));

  // Copy planes to host memory
  NvBuffer::NvBufferPlane & plane_y = buffer_->planes[0];
  NvBuffer::NvBufferPlane & plane_u = buffer_->planes[1];
  NvBuffer::NvBufferPlane & plane_v = buffer_->planes[2];
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_y.data, plane_y.fmt.stride, yuv_d_[0], yuv_step_bytes_[0], width, height,
    cudaMemcpyDeviceToHost, stream_.handle()));
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_u.data, plane_u.fmt.stride, yuv_d_[1], yuv_step_bytes_[1], width / 2, height / 2,
    cudaMemcpyDeviceToHost, stream_.handle()));
  CHECK_CUDA(cudaMemcpy2DAsync(
    plane_v.data, plane_v.fmt.stride, yuv_d_[2], yuv_step_bytes_[2], width / 2, height / 2,
    cudaMemcpyDeviceToHost, stream_.handle()));
  stream_.synchronize();

  size_t out_buf_size = width * height * 3 / 2;
  unsigned char * out_data = new unsigned char[out_buf_size];
  // NOTE: encodeFromBuffer only support YUV420
  TEST_ERROR(
    encoder_->encodeFromBuffer(buffer_.value(), JCS_YCbCr, &out_data, out_buf_size, quality),
    "NvJpeg Encoder Error");

  output->data.resize(static_cast<size_t>(out_buf_size / sizeof(uint8_t)));
  memcpy(output->data.data(), out_data, out_buf_size);

  // Free temporary data
  delete[] out_data;
  out_data = nullptr;

  return output;
}

void JetsonCompressor::set_cuda_stream(cuda::stream::handle_t & stream)
{
  stream_ = cuda::stream::wrap(
    cuda::device::current::get().id(), cuda::context::current::get().handle(), stream);
}
#endif  // JETSON_AVAILABLE

#ifdef NVJPEG_AVAILABLE
NvJpegEncoder::NvJpegEncoder()
{
  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_NVJPEG(nvjpegCreateSimple(&handle_));
  CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &state_, stream_));
  CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &params_, stream_));

  nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, stream_);

  std::memset(&nv_image_, 0, sizeof(nv_image_));
}

NvJpegEncoder::~NvJpegEncoder()
{
  CHECK_NVJPEG(nvjpegEncoderParamsDestroy(params_));
  CHECK_NVJPEG(nvjpegEncoderStateDestroy(state_));
  CHECK_NVJPEG(nvjpegDestroy(handle_));
  CHECK_CUDA(cudaStreamDestroy(stream_));
}

CompressedImage::UniquePtr NvJpegEncoder::encode(const Image & msg, int quality, ImageFormat format)
{
  CompressedImage::UniquePtr output = std::make_unique<CompressedImage>();
  output->header = msg.header;
  output->format = "jpeg";

  nvjpegEncoderParamsSetQuality(params_, quality, stream_);

  nvjpegInputFormat_t input_format;
  if (format == ImageFormat::RGB) {
    input_format = NVJPEG_INPUT_RGBI;
  } else if (format == ImageFormat::BGR) {
    input_format = NVJPEG_INPUT_BGRI;
  } else {
    std::cerr << "Specified ImageFormat is not supported" << std::endl;
  }
  setNVImage(msg);
  CHECK_NVJPEG(nvjpegEncodeImage(
    handle_, state_, params_, &nv_image_, input_format, msg.width, msg.height, stream_));

  unsigned long out_buf_size = 0;

  CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, state_, NULL, &out_buf_size, stream_));
  output->data.resize(out_buf_size);
  CHECK_NVJPEG(
    nvjpegEncodeRetrieveBitstream(handle_, state_, output->data.data(), &out_buf_size, stream_));

  CHECK_CUDA(cudaStreamSynchronize(stream_));

  return output;
}

void NvJpegEncoder::set_nv_image(const Image & msg)
{
  if (nv_image_.channel[0] == nullptr) {
    CHECK_CUDA(
      cudaMallocAsync(reinterpret_cast<void **>(&nv_image_.channel[0]), msg.data.size(), stream_));
  }

  CHECK_CUDA(cudaMemsetAsync(nv_image_.channel[0], 0, msg.data.size(), stream_));

  CHECK_CUDA(cudaMemcpyAsync(
    nv_image_.channel[0], msg.data.data(), msg.data.size(), cudaMemcpyHostToDevice, stream_));

  // int channels = image.size() / (image.width * image.height);
  int channels = 3;

  // Assuming RGBI/BGRI
  nv_image_.pitch[0] = msg.width * channels;
}

void NvJpegEncoder::set_cuda_stream(const cudaStream_t & stream)
{
  CHECK_CUDA(cudaStreamDestroy(stream_));
  stream_ = stream;
}
#endif  // NVJPEG_AVAILABLE
}  // namespace nova::compression
