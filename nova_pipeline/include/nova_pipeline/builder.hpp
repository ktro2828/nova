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

#ifndef NOVA_PIPELINE__BUILDER_HPP_
#define NOVA_PIPELINE__BUILDER_HPP_

#include "nova_pipeline/rectifier.hpp"

#include <nova_common/datatype.hpp>

#include <memory>

namespace nova::pipeline
{
/**
 * @brief Build a rectifier based on the given camera information and alpha value.
 *
 * @param info The camera information.
 * @param alpha The alpha value.
 * @return std::shared_ptr<RectifierBase> The rectifier.
 */
inline std::shared_ptr<RectifierBase> build_rectifier(const CameraInfo & info, double alpha)
{
#ifdef NPP_AVAILABLE
  return std::make_shared<NPPRectifier>(info, alpha);
#elif OPENCV_CUDA_AVAILABLE
  return std::make_shared<OpenCVRectifierGPU>(info, alpha);
#else
  return std::make_shared<OpenCVRectifierCPU>(info, alpha);
#endif
}
}  // namespace nova::pipeline
#endif  // NOVA_PIPELINE__BUILDER_HPP_
