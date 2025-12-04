cmake_minimum_required(VERSION 3.14)
project(nova_pipeline)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit)
nova_package()
autoware_package()

if(NOT OpenCV_FOUND
   AND NOT OpenCV_CUDA_FOUND
   AND NOT NPP_FOUND)
  message(FATAL_ERROR "OpenCV, OpenCV_CUDA, and NPP not found")
endif()

if(JETSON_FOUND)
  ament_auto_add_library(${PROJECT_NAME} SHARED src/rectifier.cpp)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include> ${CUDA_INCLUDE_DIR}
           ${CUDA_NPP_INCLUDES} ${OpenCV_INCLUDE_DIRS})
  target_link_directories(
    ${PROJECT_NAME} PUBLIC ${SYS_ROOT}/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
  target_link_libraries(
    ${PROJECT_NAME}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    ${OpenCV_LIBRARIES}
    CUDA::cudart
    nova_common::nova_common)
else()
  ament_auto_add_library(${PROJECT_NAME} SHARED src/rectifier.cpp)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include> ${CUDA_INCLUDE_DIR}
           ${CUDA_NPP_INCLUDES} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(
    ${PROJECT_NAME}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    ${OpenCV_LIBRARIES}
    CUDA::cudart
    nova_common::nova_common)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)

  ament_add_gtest(test_rectifier_cpu test/test_rectifier_cpu.cpp)
  target_include_directories(
    test_rectifier_cpu
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(test_rectifier_cpu ${PROJECT_NAME})
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(nova_common OpenCV)
if(NPP_AVAILABLE)
  ament_export_dependencies(nova_common CUDAToolkit)
endif()
ament_package()
