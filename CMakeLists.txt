cmake_minimum_required(VERSION 3.5)
project(buff)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(buff main.cpp)
target_link_libraries(buff ${OpenCV_LIBS})
target_link_libraries(buff "/usr/lib/libwiringPi.so") 