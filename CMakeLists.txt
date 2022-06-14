cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(EX1)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable (ex1 ex1.cpp)
target_link_libraries (ex1 ${PCL_LIBRARIES} ${OpenCV_LIBS})
