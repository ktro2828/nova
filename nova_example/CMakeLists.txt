cmake_minimum_required(VERSION 3.14)
project(nova_example)

find_package(nova_cmake REQUIRED)
find_package(autoware_cmake REQUIRED)
nova_package()
autoware_package()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
