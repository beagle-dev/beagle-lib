cmake_minimum_required(VERSION 3.9)
project(test_beagle_lib)

set(CMAKE_CXX_STANDARD 11)

# Cannot use standard library...
#set(CMAKE_CXX_FLAGS “ -stdlib=libstdc++”)

set (JAVA_HEADER_DIR "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers")

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${JAVA_HEADER_DIR}
)

# This seems the missing piece of the puzzle
link_directories(
                ${PROJECT_SOURCE_DIR}/libhmsbeagle
#        ${PROJECT_SOURCE_DIR}/cmake-build-debug/libhmsbeagle/CPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/CPU
#        ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/JNI
#        ${PROJECT_SOURCE_DIR}/libhmsbeagle/CPU/.libs
)

# This statement let cmake look for libraries in the provided directories first.
# if failed, RUNTIME_LIB = NO_DEFAULT_PATH
#FIND_LIBRARY(RUNTIME_LIB rt /usr/lib  /usr/local/lib NO_DEFAULT_PATH)

add_subdirectory(libhmsbeagle)

link_libraries(beagle_lib)

add_executable(hmctest
        examples/hmctest/hmctest.cpp)

add_executable(hmctestMedium
        examples/hmctest/hmctestMedium.cpp)

#add_executable(genomictest
#        examples/genomictest/genomictest.cpp
#        examples/genomictest/linalg.cpp
#        examples/genomictest/linalg.h
#        )
#
#add_executable(complextest
#        examples/complextest/complextest.cpp)

target_link_libraries(hmctest beagle_lib)
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest beagle_lib)