cmake_minimum_required(VERSION 3.16)
project(wood_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(draw_boxes draw_boxes.cpp)
target_link_libraries(draw_boxes PRIVATE ${OpenCV_LIBS})
target_include_directories(draw_boxes PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(wood_ai main.cpp)
target_link_libraries(wood_ai PRIVATE ${OpenCV_LIBS})
target_include_directories(wood_ai PRIVATE ${OpenCV_INCLUDE_DIRS})