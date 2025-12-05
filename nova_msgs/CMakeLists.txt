cmake_minimum_required(VERSION 3.14)
project(nova_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME} "msg/CompressedVideo.msg"
                           DEPENDENCIES builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_auto_package()
