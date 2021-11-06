cmake_minimum_required(VERSION 3.20)
project(smai)

set(CMAKE_CXX_STANDARD 20)

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(smai src/main.cpp)

target_link_libraries(smai PNG::PNG JPEG::JPEG OpenCL::OpenCL)