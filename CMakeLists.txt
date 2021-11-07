cmake_minimum_required(VERSION 3.16)
project(smai)

set(CMAKE_CXX_STANDARD 20)

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenCL REQUIRED)

file(GLOB_RECURSE sources "src/*.cpp")

add_executable(smai ${sources})

target_link_libraries(smai PNG::PNG JPEG::JPEG OpenCL::OpenCL)