cmake_minimum_required(VERSION 3.8)
project(nova_cmake)

find_package(autoware_cmake REQUIRED)
autoware_package()

list(APPEND ${PROJECT_NAME}_CONFIG_EXTRAS
  "nova_cmake-extras.cmake"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE cmake)
