cmake_minimum_required(VERSION 3.8)
project(nova_compression)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
nova_package()
autoware_package()

if(NOT JETSON
   AND NOT NVJPEG_FOUND
   AND NOT LibJpegTurbo_FOUND)
  message(FATAL_ERROR "No jpeg encoder found")
endif()

# --- JPEG encoder ---
if(JETSON)
  include(FetchContent)
  FetchContent_Declare(
    cuda-api-wrappers
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG 831666a0bfd1af0f44f4fa234ee2d983d347fcaa # v0.6.1-rc1
  )
  FetchContent_MakeAvailable(cuda-api-wrappers)

  ament_auto_add_library(
    jpeg_encoder
    SHARED
    src/jpeg_encoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp)
  target_include_directories(
    jpeg_encoder
    PUBLIC ${CUDA_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           ${SYS_ROOT}/usr/src/jetson_multimedia_api/include
           ${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b)
  target_link_directories(
    jpeg_encoder PUBLIC ${SYS_ROOT}/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
  target_link_libraries(
    jpeg_encoder ${CUDA_nppicc_LIBRARY} ${CUDA_nppidei_LIBRARY}
    $<$<BOOL:${LibJpegTurbo_FOUND}>:${LIBJPEGTURBO_LIBRARIES}> nvjpeg
    cuda-api-wrappers::runtime-and-driver)
else()
  ament_auto_add_library(jpeg_encoder SHARED src/jpeg_encoder.cpp)
  target_include_directories(
    jpeg_encoder PUBLIC ${CUDA_INCLUDE_DIR} ${CUDA_NPP_INCLUDES}
                        ${LibJpegTurbo_INCLUDE_DIRS} ${NVJPEG_INCLUDE_DIRS})
  target_link_libraries(
    jpeg_encoder
    ${CUDA_nppicc_LIBRARY}
    $<$<BOOL:${LibJpegTurbo_FOUND}>:${LibJpegTurbo_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${NVJPEG_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${CUDART_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${CULIBOS}>)
endif()

install(
  TARGETS jpeg_encoder
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_package()
