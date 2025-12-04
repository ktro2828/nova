cmake_minimum_required(VERSION 3.14)
project(nova_compression)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
nova_package()
autoware_package()

if(NOT JETSON_FOUND
   AND NOT NVJPEG_FOUND
   AND NOT LibJpegTurbo_FOUND)
  message(FATAL_ERROR "No jpeg encoder found")
endif()

# --- JPEG encoder ---
if(JETSON_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    cuda-api-wrappers
    GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
    GIT_TAG 831666a0bfd1af0f44f4fa234ee2d983d347fcaa # v0.6.1-rc1
  )
  FetchContent_MakeAvailable(cuda-api-wrappers)

  ament_auto_add_library(
    ${PROJECT_NAME}
    SHARED
    src/jpeg_encoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
           ${CUDA_INCLUDE_DIR}
           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           ${SYS_ROOT}/usr/src/jetson_multimedia_api/include
           ${SYS_ROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b)
  target_link_directories(
    ${PROJECT_NAME} PUBLIC ${SYS_ROOT}/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
  target_link_libraries(
    ${PROJECT_NAME}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    $<$<BOOL:${LibJpegTurbo_FOUND}>:${LIBJPEGTURBO_LIBRARIES}>
    nvjpeg
    cuda-api-wrappers::runtime-and-driver
    nova_common::nova_common)
else()
  ament_auto_add_library(${PROJECT_NAME} SHARED src/jpeg_encoder.cpp)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
           ${CUDA_INCLUDE_DIR}
           ${CUDA_NPP_INCLUDES}
           ${LibJpegTurbo_INCLUDE_DIRS}
           ${NVJPEG_INCLUDE_DIRS})
  target_link_libraries(
    ${PROJECT_NAME}
    ${CUDA_nppicc_LIBRARY}
    $<$<BOOL:${LibJpegTurbo_FOUND}>:${LibJpegTurbo_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${NVJPEG_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${CUDART_LIBRARY}>
    $<$<BOOL:${NVJPEG_FOUND}>:${CULIBOS}>
    nova_common::nova_common)
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)

  if(LibJpegTurbo_FOUND)
    ament_add_gtest(test_cpu_jpeg_encoder test/test_cpu_jpeg_encoder.cpp)
    target_include_directories(
      test_cpu_jpeg_encoder
      PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_link_libraries(test_cpu_jpeg_encoder ${PROJECT_NAME})
  endif()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(nova_common)
ament_package()
