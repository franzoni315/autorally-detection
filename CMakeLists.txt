cmake_minimum_required(VERSION 3.5)
project(autorally_detection)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp)

add_executable(main ${SOURCE_FILES})
target_link_libraries( main ${OpenCV_LIBS} )