cmake_minimum_required(VERSION 3.4)
project(autorally_detection)

#set(CMAKE_PREFIX_PATH "/home/igor/Documents/opencv-2.4.13/build")
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(main main.cpp)
target_link_libraries( main ${OpenCV_LIBS})

add_executable(viz viz.cpp)
target_link_libraries( viz ${OpenCV_LIBS})