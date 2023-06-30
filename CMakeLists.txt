cmake_minimum_required(VERSION 3.9)
project(BEAGLE)

set(BEAGLE_PACKAGE_NAME "libhmsbeagle")
set(BEAGLE_VERSION_MAJOR "4")
set(BEAGLE_VERSION_MINOR "0")
set(BEAGLE_VERSION_PATCH "1")

set(BEAGLE_VERSION "${BEAGLE_VERSION_MAJOR}.${BEAGLE_VERSION_MINOR}.${BEAGLE_VERSION_PATCH}")
set(BEAGLE_PLUGIN_VERSION "${BEAGLE_VERSION_MAJOR}${BEAGLE_VERSION_MINOR}")

set(BEAGLE_OPTIMIZE_FOR_NATIVE_ARCH true)

option(BUILD_OPENCL "Build beagle with OpenCL library" ON)
option(BUILD_CUDA "Build beagle with CUDA library" ON)
option(BUILD_JNI "Build beagle with JNI library" ON)
option(BUILD_OPENMP "Build beagle with OpenMP library" ON)
option(BUILD_SSE "Build beagle with SSE" ON)

# Old config.h settings

OPTION(BEAGLE_BENCHMARK "Print runtimes" OFF)
OPTION(BEAGLE_DEBUG_FLOW "Print flow" OFF)
OPTION(BEAGLE_DEBUG_VALUES "Print debug values" OFF)
OPTION(BEAGLE_DEBUG_SYNCH "Synchronize threads on GPU for timing" OFF)
OPTION(BEAGLE_TENSOR_CORES "Compile with support for tensor cores" OFF)
OPTION(BEAGLE_BENCHMARK_ENERGY "Print energy consumption" OFF)

IF(BEAGLE_BENCHMARK)
    ADD_DEFINITIONS(-DBEAGLE_BENCHMARK)
ENDIF(BEAGLE_BENCHMARK)

IF(BEAGLE_BENCHMARK_ENERGY)
	ADD_DEFINITIONS(-DBEAGLE_BENCHMARK_ENERGY)
ENDIF(BEAGLE_BENCHMARK_ENERGY)

IF(BEAGLE_DEBUG_FLOW)
    ADD_DEFINITIONS(-DBEAGLE_DEBUG_FLOW)
ENDIF(BEAGLE_DEBUG_FLOW)

IF(BEAGLE_DEBUG_VALUES)
    ADD_DEFINITIONS(-DBEAGLE_DEBUG_VALUES)
ENDIF(BEAGLE_DEBUG_VALUES)

IF(BEAGLE_TENSOR_CORES)
	ADD_DEFINITIONS(-DBEAGLE_TENSOR_CORES)
ENDIF(BEAGLE_TENSOR_CORES)

IF(BEAGLE_DEBUG_SYNCH)
	ADD_DEFINITIONS(-DBEAGLE_DEBUG_SYNCH)
ENDIF(BEAGLE_DEBUG_SYNCH)

add_definitions(
	-DPACKAGE_BUGREPORT="beagle-dev@googlegroups.com"
	-DPACKAGE_NAME="${BEAGLE_PACKAGE_NAME}"
	-DPACKAGE_STRING="${BEAGLE_PACKAGE_NAME} ${BEAGLE_VERSION}"
	-DPACKAGE_TARNAME="${BEAGLE_PACKAGE_NAME}"
	-DPACKAGE_VERSION="${BEAGLE_VERSION}"
	-DPACKAGE_URL=""
	-DPLUGIN_VERSION="${BEAGLE_PLUGIN_VERSION}"
	)

set(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

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

#if(NOT WIN32)
#    add_definitions(-DHAVE_CONFIG_H)
    if(APPLE)
        add_definitions(-DDLS_MACOS)
    endif(APPLE)
#endif(NOT WIN32)

include_directories(
        ${PROJECT_SOURCE_DIR}
)

find_package(JNI)
if(JNI_FOUND)
    message (STATUS "JAVA_HOME=$ENV{JAVA_HOME}")
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")

	include_directories(
    	    $ENV{JAVA_HOME}/include
	        ${JNI_INCLUDE_DIRS}
	)
else(JNI_FOUND)
	if(BUILD_JNI)
		message (FATAL_ERROR "No JNI includes and libraries found")
	endif(BUILD_JNI)
endif(JNI_FOUND)

if(BUILD_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		# Do we need to do anything?  Probably need to statically link omp to cpu *.so
	else(OpenMP_CXX_FOUND)
		message (FATAL_ERROR "No OpenMP Found")
	endif(OpenMP_CXX_FOUND)
endif(BUILD_OPENMP)

link_directories(
        ${PROJECT_SOURCE_DIR}/libhmsbeagle
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/CPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU
        ${PROJECT_SOURCE_DIR}/libhmsbeagle/JNI
)

# This statement let cmake look for libraries in the provided directories first.
# if failed, RUNTIME_LIB = NO_DEFAULT_PATH
#FIND_LIBRARY(RUNTIME_LIB rt /usr/lib  /usr/local/lib NO_DEFAULT_PATH)

if(APPLE)
#  	execute_process(
#		COMMAND uname -m
#		RESULT_VARIABLE result
#    	OUTPUT_VARIABLE BEAGLE_OSX_NATIVE_ARCHITECTURE
#    	OUTPUT_STRIP_TRAILING_WHITESPACE#
#	)

	# Check if running under rosetta on arm64 hardware
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
        execute_process(COMMAND sysctl -q hw.optional.arm64
        	OUTPUT_VARIABLE _sysctl_stdout
            ERROR_VARIABLE _sysctl_stderr
            RESULT_VARIABLE _sysctl_result
        )
        if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
            set(CMAKE_HOST_SYSTEM_PROCESSOR "arm64")
        endif()
    endif()
	set(BEAGLE_OSX_NATIVE_ARCHITECTURE ${CMAKE_HOST_SYSTEM_PROCESSOR})

	if(BEAGLE_OSX_NATIVE_ARCHITECTURE STREQUAL "arm64")
		set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
		message(STATUS "macOS universal (x86_64 / arm64) build")
	else()
		message(STATUS "macOS x86_64 build")
	endif()

	# Seems necessary from macOS 10.15
	SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif(APPLE)

if (BEAGLE_OPTIMIZE_FOR_NATIVE_ARCH)
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
	if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
	else()
		CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
		if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
		endif()
	endif()
endif()

add_subdirectory(libhmsbeagle)

link_libraries(hmsbeagle)

enable_testing()

add_subdirectory(examples)

set(CPACK_PACKAGE_VERSION "${BEAGLE_VERSION}")
set(CPACK_PACKAGE_MAJOR "${BEAGLE_VERSION_MAJOR}")
set(CPACK_PACKAGE_MINOR "${BEAGLE_VERSION_MINOR}")
set(CPACK_PACKAGE_PATCH "${BEAGLE_VERSION_PATCH}")

if(WIN32)
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
	set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

    set(CPACK_GENERATOR WIX)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Common Files/libhmsbeagle")
    set(CPACK_WIX_UPGRADE_GUID "57FC5933-B420-4F10-97F3-B5DDCFF2AE13")
    set(CPACK_WIX_LICENSE_RTF "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/LICENSE.rtf")
    set(CPACK_WIX_UI_BANNER "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/banner.jpg")
    set(CPACK_WIX_UI_DIALOG "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")
    set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/patch.xml")
endif(WIN32)

if(APPLE)
	set(CPACK_GENERATOR productbuild)
	set(CPACK_BUNDLE_NAME "${BEAGLE_PACKAGE_NAME}")
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/LICENSE.rtf")
	set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/project/beagle-macos/README.txt")

	set(CPACK_PRODUCTBUILD_BACKGROUND_UTI "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")

	get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)

	include(CPackComponent)

	cpack_add_component(main
			DISPLAY_NAME "Main library"
			DESCRIPTION "BEAGLE main library"
			REQUIRED)

	cpack_add_component(cpu
  			DISPLAY_NAME "CPU plugin"
		  	DESCRIPTION "CPU plugin"
  			REQUIRED)

  	cpack_add_component(cpu_sse
  			DISPLAY_NAME "CPU-SSE plugin"
  			DESCRIPTION "CPU-SSE plugin")

  	cpack_add_component(cuda
  			DISPLAY_NAME "CUDA plugin"
  			DESCRIPTION "CUDA plugin")

  	cpack_add_component(opencl
  			DISPLAY_NAME "OpenCL plugin"
  			DESCRIPTION "OpenCL plugin")

	set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
endif(APPLE)

include(CPack)
