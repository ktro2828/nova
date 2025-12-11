cmake_minimum_required(VERSION 3.14)
project(nova_example)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
find_package(CUDAToolkit)
nova_package()
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED src/talker.cpp src/listener.cpp)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(
  ${PROJECT_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}
  nova_compression::nova_compression nova_pipeline::nova_pipeline
  nova_ros::nova_ros)

rclcpp_components_register_node(${PROJECT_NAME} PLUGIN "nova::example::Talker"
                                EXECUTABLE ${PROJECT_NAME}_talker)
rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "nova::example::Listener" EXECUTABLE
  ${PROJECT_NAME}_listener)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_package()
