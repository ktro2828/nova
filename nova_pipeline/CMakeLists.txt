cmake_minimum_required(VERSION 3.14)
project(nova_pipeline)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
find_package(OpenCV REQUIRED)
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
    nova_common::nova_common)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(nova_common)
ament_package()
