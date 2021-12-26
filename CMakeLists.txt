cmake_minimum_required(VERSION 3.21)
project(sycl_lb)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++20")

find_package(PNG REQUIRED)

add_executable(sycl_lb main.cpp)
target_link_libraries(sycl_lb PNG::PNG)
