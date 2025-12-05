// -*-c++-*---------------------------------------------------------------------------------------
// Copyright 2024 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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

#ifndef NOVA_ROS__UTILITY_HPP_
#define NOVA_ROS__UTILITY_HPP_

#include <ffmpeg_encoder_decoder/utils.hpp>

#include <string>
#include <vector>

namespace nova::ros::utility
{
/**
 * @brief Splits a string of comma-separated options into a vector of strings.
 *
 * @param options The string containing comma-separated options.
 * @return std::vector<std::string> A vector of strings, each representing an option.
 */
inline std::vector<std::string> split_av_options(const std::string & options)
{
  return ffmpeg_encoder_decoder::utils::split_by_char(options, ',');
}

/**
 * @brief Splits a string of colon-separated options into a vector of strings.
 *
 * @param option The string containing colon-separated options.
 * @return std::vector<std::string> A vector of strings, each representing an option.
 */
inline std::vector<std::string> split_av_option(const std::string & option)
{
  return ffmpeg_encoder_decoder::utils::split_by_char(option, ':');
}

}  // namespace nova::ros::utility
#endif  // NOVA_ROS__UTILITY_HPP_
