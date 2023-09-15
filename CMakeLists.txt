cmake_minimum_required(VERSION 3.13)

set(PRJ_NAME samples_tests)
project(samples_tests)

find_package(TBB REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(/opt/intel/oneapi/tbb/latest/include/ ./)
add_executable(${PRJ_NAME} main.cpp test_parallel_reduce.cpp)
target_link_libraries(${PRJ_NAME} TBB::tbb)
