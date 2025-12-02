# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

macro(nova_package)
  # Ensure local CMake modules (FindLibJpegTurbo.cmake, FindNVJPEG.cmake) are
  # discoverable
  list(APPEND CMAKE_MODULE_PATH "${nova_cmake_DIR}")

  # --- Try to find JpegTurbo ---
  find_package(LibJpegTurbo)
  if(LibJpegTurbo_FOUND)
    add_definitions(-DTURBOJPEG_AVAILABLE)
  else()
    message(WARNING "[JpegTurbo] Not found")
  endif()

  # --- Try to find NVJPEG dependencies ---
  find_package(NVJPEG)
  find_package(CUDA)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  if(NVJPEG_FOUND
     AND CUDART_LIBRARY
     AND CULIBOS)
    add_definitions(-DNVJPEG_AVAILABLE)
  else()
    message(WARNING "[NVJPEG] Not found")
  endif()

  # --- Try to find NPP libraries ---
  find_library(CUDA_nppicc_LIBRARY nppicc
               ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUDA_nppidei_LIBRARY nppidei
               ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUDA_nppig_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(CUDA_nppisu_LIBRARY nppisu
               ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  if(CUDA_nppicc_LIBRARY
     AND CUDA_nppidei_LIBRARY
     AND CUDA_nppig_LIBRARY
     AND CUDA_nppisu_LIBRARY)
    set(NPP_FOUND TRUE)
    add_definitions(-DNPP_AVAILABLE)
  else()
    set(NPP_FOUND FALSE)
    message(WARNING "[NPP] Not found")
  endif()

  # --- Try to find OpenCV ---
  if(OpenCV_FOUND)
    find_package(cv_bridge REQUIRED)
    add_definitions(-DOPENCV_AVAILABLE)
  else()
    message(WARNING "[OpenCV] Not found")
  endif()

  # --- Try to find OpenCV with CUDA ---
  if(OpenCV_CUDA_VERSION)
    set(OpenCV_CUDA_FOUND TRUE)
    add_definitions(-DOPENCV_CUDA_AVAILABLE)
  else()
    set(OpenCV_CUDA_FOUND FALSE)
    message(WARNING "[OpenCV CUDA] Not found")
  endif()

  # --- Try to find JETSON environment ---
  if(EXISTS "/etc/nv_tegra_release")
    set(JETSON_FOUND TRUE)
    add_definitions(-DJETSON_AVAILABLE)
  else()
    set(JETSON_FOUND FALSE)
    message(WARNING "[Jetson] Not found")
  endif()
endmacro()
