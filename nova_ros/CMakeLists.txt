cmake_minimum_required(VERSION 3.14)
project(nova_ros)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
nova_package()
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED src/publisher.cpp)

target_link_libraries(
  ${PROJECT_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}
  nova_pipeline::nova_pipeline nova_compression::nova_compression)

rclcpp_components_register_node(${PROJECT_NAME} PLUGIN "nova::ros::Publisher"
                                EXECUTABLE ${PROJECT_NAME}_publisher)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(nova_pipeline nova_compression)
ament_package()
