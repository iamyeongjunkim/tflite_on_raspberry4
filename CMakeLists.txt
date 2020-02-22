cmake_minimum_required(VERSION 3.15)
project(tfliteDriver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(/usr/local/Cellar/flatbuffers/1.11.0/include)

add_executable(tfliteDriver main.cpp NNModelParser.cpp NNModelParser.h errorType.h)