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

#ifndef NOVA_COMMON__FORMAT_HPP_
#define NOVA_COMMON__FORMAT_HPP_

#include <cstdint>

namespace nova
{
/**
 * @brief Image format enum class.
 */
enum class ImageFormat : uint8_t { RGB = 0, BGR = 1 };
}  // namespace nova
#endif  // NOVA_COMMON__FORMAT_HPP_
