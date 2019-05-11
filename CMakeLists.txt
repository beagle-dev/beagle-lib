cmake_minimum_required(VERSION 3.9)
project(test_beagle_lib)

set(CMAKE_CXX_STANDARD 11)

## flags for standard library
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(JNI REQUIRED)
if(JNI_FOUND)
    message (STATUS "JAVA_HOME=$ENV{JAVA_HOME}")
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif(JNI_FOUND)

include_directories(
        $ENV{JAVA_HOME}/include
        ${PROJECT_SOURCE_DIR}
        ${JNI_INCLUDE_DIRS}
)

link_directories(
        ${PROJECT_SOURCE_DIR}/libhmsbeagle
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/CPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/JNI
)

# This statement let cmake look for libraries in the provided directories first.
# if failed, RUNTIME_LIB = NO_DEFAULT_PATH
#FIND_LIBRARY(RUNTIME_LIB rt /usr/lib  /usr/local/lib NO_DEFAULT_PATH)

add_subdirectory(libhmsbeagle)

link_libraries(hmsbeagle)

add_executable(hmctest
        examples/hmctest/hmctest.cpp)

add_executable(tinytest
        examples/tinytest/tinytest.cpp)

#add_executable(hmctestMedium
#        examples/hmctest/hmctestMedium.cpp)

#add_executable(genomictest
#        examples/genomictest/genomictest.cpp
#        examples/genomictest/linalg.cpp
#        examples/genomictest/linalg.h
#        )
#
#add_executable(complextest
#        examples/complextest/complextest.cpp)

#target_link_libraries(hmctest hmsbeagle hmsbeagle-opencl)
#target_link_libraries(genomictest beagle_lib)
target_link_libraries(tinytest hmsbeagle)
