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

#ifndef NOVA_COMPRESSION__BUILDER_HPP_
#define NOVA_COMPRESSION__BUILDER_HPP_

#include "nova_compression/jpeg_encoder.hpp"

#include <memory>
#include <string>

namespace nova::compression
{
/**
 * Build a JPEG encoder.
 *
 * @param name The name of the encoder.
 * @return A shared pointer to the encoder.
 */
inline std::shared_ptr<JpegEncoderBase> build_jpeg_encoder(std::string name)
{
#if JETSON_AVAILABLE
  return std::make_shared<JetsonJpegEncoder>(name);
#elif NVJPEG_AVAILABLE
  return std::make_shared<NvJpegEncoder>(name);
#elif TURBOJPEG_AVAILABLE
  return std::make_shared<CpuJpegEncoder>(name);
#else
  throw std::runtime_error("No JPEG encoder available");
#endif
}
}  // namespace nova::compression
#endif  // NOVA_COMPRESSION__BUILDER_HPP_
