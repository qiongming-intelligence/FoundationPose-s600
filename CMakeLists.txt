cmake_minimum_required(VERSION 3.16)

project(foundationpose_s600 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FOUNDATIONPOSE_S600_BUILD_APPS "Build FoundationPose S600 command line applications" ON)
option(FOUNDATIONPOSE_S600_ENABLE_HOBOT "Enable D-Robotics/Hobot runtime integration" ON)

add_subdirectory(src/csrc)
