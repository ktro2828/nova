cmake_minimum_required(VERSION 3.14)
project(nova_ros)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
find_package(CUDAToolkit)
nova_package()
autoware_package()

if(${image_transport_VERSION} VERSION_GREATER_EQUAL "5.0.0")
  add_definitions(-DIMAGE_TRANSPORT_USE_PUBLISHER_T)
  add_definitions(-DIMAGE_TRANSPORT_NEEDS_PUBLISHEROPTIONS)
endif()
if(${image_transport_VERSION} VERSION_GREATER_EQUAL "6.0.0")
  add_definitions(-DIMAGE_TRANSPORT_RESOLVES_BASE_TOPIC)
endif()
if(${image_transport_VERSION} VERSION_GREATER_EQUAL "6.3.0")
  add_definitions(-DIMAGE_TRANSPORT_USE_QOS)
endif()
if(${image_transport_VERSION} VERSION_GREATER_EQUAL "6.4.0")
  add_definitions(-DIMAGE_TRANSPORT_USE_NODEINTERFACE)
endif()

ament_auto_add_library(${PROJECT_NAME} SHARED src/imgproc_node.cpp src/qos.cpp
                       src/publisher.cpp src/subscriber.cpp)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(
  ${PROJECT_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}
  nova_pipeline::nova_pipeline nova_compression::nova_compression)

rclcpp_components_register_node(${PROJECT_NAME} PLUGIN "nova::ros::ImgProcNode"
                                EXECUTABLE ${PROJECT_NAME}_imgproc_node)

pluginlib_export_plugin_description_file(${PROJECT_NAME} plugin_description.xml)

set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(nova_pipeline nova_compression)
ament_auto_package(INSTALL_TO_SHARE config USE_SCOPED_HEADER_INSTALL_DIR)
