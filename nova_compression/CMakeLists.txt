cmake_minimum_required(VERSION 3.8)
project(nova_compression)

find_package(nova_cmake REQUIRED)
nova_package()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
