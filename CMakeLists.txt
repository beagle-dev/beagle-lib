cmake_minimum_required(VERSION 3.9)
project(BEAGLE)

set(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(BEAGLE_VERSION_MAJOR "3")
set(BEAGLE_VERSION_MINOR "2")
set(BEAGLE_VERSION_PATCH "0")

set(BEAGLE_VERSION "${BEAGLE_VERSION_MAJOR}.${BEAGLE_VERSION_MINOR}.${BEAGLE_VERSION_PATCH}")
set(BEAGLE_PLUGIN_VERSION "${BEAGLE_VERSION_MAJOR}${BEAGLE_VERSION_MINOR}")

## flags for standard library
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /Ot /MT")
else(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread")
	#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -pthread")
	#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
endif(WIN32)

if(NOT WIN32)
    add_definitions(-DHAVE_CONFIG_H)
    if(APPLE)
        add_definitions(-DDLS_MACOS)
    endif(APPLE)
endif(NOT WIN32)

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

add_executable(hmctest5
		examples/hmctest/hmctest5.cpp)

add_executable(hmcGaptest
        examples/hmctest/hmcGapTest.cpp)

#add_executable(tinytest
#        examples/tinytest/tinytest.cpp)

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

target_link_libraries(hmctest hmsbeagle ${CMAKE_DL_LIBS})
target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)

set(CPACK_PACKAGE_VERSION "${BEAGLE_VERSION}")
set(CPACK_PACKAGE_MAJOR "${BEAGLE_VERSION_MAJOR}")
set(CPACK_PACKAGE_MINOR "${BEAGLE_VERSION_MINOR}")
set(CPACK_PACKAGE_PATCH "${BEAGLE_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

if(WIN32) 
    set(CPACK_GENERATOR WIX)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Common Files/libhmsbeagle")
    set(CPACK_WIX_UPGRADE_GUID "57FC5933-B420-4F10-97F3-B5DDCFF2AE13")
    set(CPACK_WIX_LICENSE_RTF "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/COPYING.LESSER.rtf")
    set(CPACK_WIX_UI_BANNER "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/banner.jpg")
    set(CPACK_WIX_UI_DIALOG "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")
    set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/patch.xml")
endif(WIN32)

include(CPack)
