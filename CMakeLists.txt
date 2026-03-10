cmake_minimum_required(VERSION 3.16)
project(wood_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT is not set")
endif()

add_executable(wood_ai main.cpp)

target_include_directories(wood_ai PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(wood_ai PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(wood_ai PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)