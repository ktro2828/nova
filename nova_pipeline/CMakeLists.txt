cmake_minimum_required(VERSION 3.8)
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
  ament_auto_add_library(rectifier SHARED src/rectifier.cpp)
  target_include_directories(rectifier PUBLIC ${CUDA_INCLUDE_DIR}
                                              ${OpenCV_INCLUDE_DIRS})
  target_link_directories(
    rectifier PUBLIC ${SYS_ROOT}/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu
    ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/tegra)
  target_link_libraries(
    rectifier ${CUDA_nppidei_LIBRARY} ${CUDA_nppig_LIBRARY}
    ${CUDA_nppicc_LIBRARY} ${CUDA_nppisu_LIBRARY} ${OpenCV_LIBRARIES})
else()
  ament_auto_add_library(rectifier SHARED src/rectifier.cpp)
  target_include_directories(
    rectifier PUBLIC ${CUDA_INCLUDE_DIR} ${CUDA_NPP_INCLUDES}
                     ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(
    rectifier ${CUDA_nppidei_LIBRARY} ${CUDA_nppig_LIBRARY}
    ${CUDA_nppicc_LIBRARY} ${CUDA_nppisu_LIBRARY} ${OpenCV_LIBRARIES})
endif()

install(
  TARGETS rectifier
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_package()
