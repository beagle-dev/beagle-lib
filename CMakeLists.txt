cmake_minimum_required(VERSION 3.18)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

project(BEAGLE)

set(BEAGLE_PACKAGE_NAME "libhmsbeagle")
set(BEAGLE_VERSION_MAJOR "4")
set(BEAGLE_VERSION_MINOR "3")
set(BEAGLE_VERSION_PATCH "0")

set(GENERIC_API_VERSION "1")

set(BEAGLE_VERSION "${BEAGLE_VERSION_MAJOR}.${BEAGLE_VERSION_MINOR}.${BEAGLE_VERSION_PATCH}")
set(BEAGLE_PLUGIN_VERSION "${BEAGLE_VERSION_MAJOR}${BEAGLE_VERSION_MINOR}")

message(STATUS "BEAGLE_VERSION = ${BEAGLE_VERSION}")
message(STATUS "BEAGLE_PLUGIN_VERSION = ${BEAGLE_PLUGIN_VERSION}")

option(BEAGLE_OPTIMIZE_FOR_NATIVE_ARCH "Compile for the build machine's CPU (-march=native); turn off for binaries that run on other machines, such as a CI cache" ON)

set(BEAGLE_CPACK_PACKAGE_NAME "BEAGLE")
set(BASTA_EXTENSION "BIT")

option(BUILD_OPENCL "Build beagle with OpenCL library" ON)
option(BUILD_CUDA "Build beagle with CUDA library" ON)
# TinyGPU (macOS only: TinyGPU.app, IOKit): on by default where its kernel tools (nvcc, ptxas; libhmsbeagle/GPU/CMakeLists.txt's
# lookups) are found, so a plain build elsewhere skips it; -DBUILD_TINYGPU=ON requires them
if(NOT DEFINED BUILD_TINYGPU AND APPLE)
	find_program(TINYGPU_NVCC NAMES nvcc HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	find_program(TINYGPU_PTXAS NAMES ptxas HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
endif()
if(APPLE AND TINYGPU_NVCC AND TINYGPU_PTXAS)
	set(BUILD_TINYGPU_DEFAULT ON)
else()
	set(BUILD_TINYGPU_DEFAULT OFF)
endif()
option(BUILD_TINYGPU "Build beagle with TinyGPU library (an eGPU through TinyGPU.app, booted in C++)" ${BUILD_TINYGPU_DEFAULT})
if(NOT BUILD_TINYGPU)
	message(STATUS "TinyGPU: not built (macOS with nvcc and ptxas only; -DBUILD_TINYGPU=ON requires them)")
endif()
option(BUILD_JNI "Build beagle with JNI library" ON)
option(BUILD_OPENMP "Build beagle with OpenMP library" OFF)
option(BUILD_ACTION "Build beagle with action-based calculations" ON)
option(BUILD_SSE "Build beagle with SSE library" ON)
option(BUILD_SPECTRAL "Build beagle with spectral representation" ON)
option(BUILD_FUSED "TEST fused post-order implementation (hmsbeagle-opencl-fused; fused kernel in TinyGPU modules)" OFF)
option(BUILD_HIERARCHICAL "Build beagle with hierarchical representation" ON)
option(BUILD_BIT "Build beagle with basta library" OFF)
# MASCOT builds with clang, GCC and MSVC (MSVC: kernel A only)
option(BUILD_MASCOT "Build the MASCOT structured-coalescent component" ON)

# Old config.h settings

OPTION(BEAGLE_BENCHMARK "Print runtimes" OFF)
OPTION(BEAGLE_DEBUG_FLOW "Print flow" OFF)
OPTION(BEAGLE_DEBUG_VALUES "Print debug values" OFF)
OPTION(BEAGLE_DEBUG_SYNCH "Synchronize threads on GPU for timing" OFF)
OPTION(BEAGLE_TENSOR_CORES "Compile with support for tensor cores" OFF)
OPTION(BEAGLE_BENCHMARK_ENERGY "Print energy consumption" OFF)
OPTION(BEAGLE_BASTA_CPU_ACCUM "Use CPU for final accumulation for reduction" OFF)
OPTION(BEAGLE_INDEX_CHECKS "Range-check partials, matrix and scale indices on the CPU (BEAGLE_ERROR_OUT_OF_RANGE); for CI and testing" OFF)
OPTION(BEAGLE_DEBUG_MEMORY "Count GPU buffers, sub-buffers and device bytes per instance and report them on stderr; OpenCL also honours BEAGLE_DEBUG_OPENCL_ALIGN" OFF)

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

IF(BEAGLE_BASTA_CPU_ACCUM)
	ADD_DEFINITIONS(-DBEAGLE_BASTA_CPU_ACCUM)
ENDIF(BEAGLE_BASTA_CPU_ACCUM)

# a compile definition, so it applies whatever CMAKE_BUILD_TYPE is
IF(BEAGLE_INDEX_CHECKS)
	ADD_DEFINITIONS(-DBEAGLE_INDEX_CHECKS)
ENDIF(BEAGLE_INDEX_CHECKS)

IF(BEAGLE_DEBUG_MEMORY)
	ADD_DEFINITIONS(-DBEAGLE_DEBUG_MEMORY)
ENDIF(BEAGLE_DEBUG_MEMORY)

IF(BEAGLE_DEBUG_LOAD)
	ADD_DEFINITIONS(-DBEAGLE_DEBUG_LOAD)
ENDIF(BEAGLE_DEBUG_LOAD)

add_definitions(
	-DPACKAGE_BUGREPORT="beagle-dev@googlegroups.com"
	-DPACKAGE_NAME="${BEAGLE_PACKAGE_NAME}"
	-DPACKAGE_STRING="${BEAGLE_PACKAGE_NAME} ${BEAGLE_VERSION}"
	-DPACKAGE_TARNAME="${BEAGLE_PACKAGE_NAME}"
	-DPACKAGE_VERSION="${BEAGLE_VERSION}"
	-DPACKAGE_URL=""
	-DPLUGIN_VERSION="${BEAGLE_PLUGIN_VERSION}"
	)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Make erros and warnings more readable by enabling colors in the compiler output
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
endif()

## flags for standard library
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
else(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
	# the -O2 of RelWithDebInfo comes after CMAKE_CXX_FLAGS on the command line and would override the -O3 below
	string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif(WIN32)

if (NOT "${CMAKE_BUILD_TYPE}" MATCHES Debug)
        if(WIN32)
	        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /Ot")
        else(WIN32)
	        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
        endif(WIN32)
endif()

#if(NOT WIN32)
#    add_definitions(-DHAVE_CONFIG_H)
    if(APPLE)
        add_definitions(-DDLS_MACOS)
    endif(APPLE)
#endif(NOT WIN32)

include_directories(
        ${PROJECT_SOURCE_DIR}
)

if(BUILD_JNI)
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
		message (FATAL_ERROR "No JNI includes and libraries found")
	endif(JNI_FOUND)
endif(BUILD_JNI)

if(BUILD_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		# Do we need to do anything?  Probably need to statically link omp to cpu *.so
	else(OpenMP_CXX_FOUND)
		message (FATAL_ERROR "No OpenMP Found")
	endif(OpenMP_CXX_FOUND)
endif(BUILD_OPENMP)

if(BUILD_ACTION)
        include(FetchContent)
        set(EIGEN_VERSION 3.4)
        find_package(Eigen3 ${EIGEN_VERSION} QUIET)
        if(NOT Eigen3_FOUND)
                message(STATUS "Eigen3 not found — downloading.")

                set(BUILD_TESTING OFF CACHE INTERNAL "")

                FetchContent_Declare(
                        eigen
                        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                        GIT_TAG        ${EIGEN_VERSION}
                        GIT_SHALLOW    ON
                        SOURCE_SUBDIR  "NONEXISTANT_DIR"
                )

                FetchContent_MakeAvailable(eigen)
                message(STATUS "   Eigen3: downloaded.")

                unset(BUILD_TESTING CACHE)

                # Create our own Eigen3::Eigen target instead of running the CMakeLists.txt from Eigen,
                #  which is creates an uninstallable target and so fails to configure.
                add_library(Eigen3::Eigen INTERFACE IMPORTED)
                target_include_directories(Eigen3::Eigen INTERFACE
                        ${eigen_SOURCE_DIR}
                )
        endif()

	set(CMAKE_CXX_STANDARD 17)
	add_definitions(-DBEAGLE_ACTION)
#	add_definitions(-DBEAGLE_DEBUG_FLOW)
endif(BUILD_ACTION)

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
elseif(UNIX)
        # This tells libhmsbeagle.so to search the same directory to find plugins.
        SET(CMAKE_INSTALL_RPATH "\$ORIGIN")
        # $ORIGIN/{CPU,GPU,JNI} is so that libhmsbeagle.so can find the plugins in their build location.
        # $ORIGIN/../libhmsbeagle is so that examples/* can find libhmsbeagle.so
	SET(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/CPU:\$ORIGIN/GPU:\$ORIGIN/GPU/CMake_Action:\$ORIGIN/GPU/CMake_CUDA:\$ORIGIN/GPU/CMake_OpenCL:\$ORIGIN/JNI:\$ORIGIN/../libhmsbeagle")
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

# MASCOT's golden tests and tests/opencl (added by libhmsbeagle/GPU) are registered from inside libhmsbeagle
enable_testing()
add_subdirectory(libhmsbeagle)

link_libraries(hmsbeagle)

enable_testing()

add_subdirectory(examples)

IF(BEAGLE_DEBUG_KERNELS)
    add_subdirectory(tests/tensor_cores)
    add_subdirectory(tests/basta)
ENDIF(BEAGLE_DEBUG_KERNELS)

set(CPACK_PACKAGE_VERSION "${BEAGLE_VERSION}")
set(CPACK_PACKAGE_MAJOR "${BEAGLE_VERSION_MAJOR}")
set(CPACK_PACKAGE_MINOR "${BEAGLE_VERSION_MINOR}")
set(CPACK_PACKAGE_PATCH "${BEAGLE_VERSION_PATCH}")

if(BUILD_BIT)
	set(CPACK_PACKAGE_NAME "${BEAGLE_CPACK_PACKAGE_NAME}-${BASTA_EXTENSION}")
else(BUILD_BIT)
	set(CPACK_PACKAGE_NAME "${BEAGLE_CPACK_PACKAGE_NAME}")
endif(BUILD_BIT)

message(STATUS "CPack package name: ${CPACK_PACKAGE_NAME}")

if(WIN32)
	set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
	set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

    set(CPACK_GENERATOR WIX)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Common Files/libhmsbeagle")
    set(CPACK_WIX_UPGRADE_GUID "57FC5933-B420-4F10-97F3-B5DDCFF2AE13")
    set(CPACK_WIX_LICENSE_RTF "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/LICENSE.rtf")
    set(CPACK_WIX_UI_BANNER "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/banner.jpg")
    set(CPACK_WIX_UI_DIALOG "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/welcome.jpg")
    set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/project/beagle-vs-2019-wix/patch.xml") # MAS commented out for testing

# 	get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
#
# 	include(CPackComponent)
#
# 	cpack_add_component(main
# 			DISPLAY_NAME "Main library"
# 			DESCRIPTION "BEAGLE main library"
# 			REQUIRED)
#
# 	cpack_add_component(cpu
#   			DISPLAY_NAME "CPU plugin"
# 		  	DESCRIPTION "CPU plugin"
#   			REQUIRED)
#
#   	cpack_add_component(cpu_sse
#   			DISPLAY_NAME "CPU-SSE plugin"
#   			DESCRIPTION "CPU-SSE plugin")
#
#   	cpack_add_component(cpu_omp
#   			DISPLAY_NAME "CPU-OMP plugin"
#   			DESCRIPTION "CPU-OMP plugin")
#
#   	cpack_add_component(jni
#   			DISPLAY_NAME "JNI plugin"
#   			DESCRIPTION "JNI plugin")
#
#   	cpack_add_component(jni_basta
#   			DISPLAY_NAME "JNI-${BASTA_EXTENSION} plugin"
#   			DESCRIPTION "JNI-${BASTA_EXTENSION} plugin")
#
#   	cpack_add_component(cuda
#   			DISPLAY_NAME "CUDA plugin"
#   			DESCRIPTION "CUDA plugin")
#
#   	cpack_add_component(opencl
#   			DISPLAY_NAME "OpenCL plugin"
#   			DESCRIPTION "OpenCL plugin")
#
# # Install files directly to the root
# install(FILES my_file.txt DESTINATION "/")

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

	cpack_add_component(cpu_action
			DISPLAY_NAME "CPU-Action plugin"
			DESCRIPTION "CPU-Action plugin")

  	cpack_add_component(cpu_omp
  			DISPLAY_NAME "CPU-OMP plugin"
  			DESCRIPTION "CPU-OMP plugin")

  	cpack_add_component(cpu_hierarchical
  			DISPLAY_NAME "CPU-Hierarchical plugin"
  			DESCRIPTION "CPU plugin for hierarchical state-tree substitution models")

  	cpack_add_component(jni
  			DISPLAY_NAME "JNI plugin"
  			DESCRIPTION "JNI plugin")

  	cpack_add_component(jni_basta
  			DISPLAY_NAME "JNI-${BASTA_EXTENSION} plugin"
  			DESCRIPTION "JNI-${BASTA_EXTENSION} plugin")

  	cpack_add_component(cuda
  			DISPLAY_NAME "CUDA plugin"
  			DESCRIPTION "CUDA plugin")

	cpack_add_component(cuda_tensor_cores
			DISPLAY_NAME "Tensor Cores plugin"
			DESCRIPTION "Tensor Cores plugin")

  	cpack_add_component(opencl
  			DISPLAY_NAME "OpenCL plugin"
  			DESCRIPTION "OpenCL plugin")

	set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

	set(CPACK_BUNDLE_APPLE_CERT_APP "Developer ID Applicaton: Marc Suchard (LC992Z5R7E)")
endif(APPLE)

include(CPack)

if(APPLE)
#         <exec executable="codesign">
#             <arg value="-s"/>
#             <arg value="${AppleSigner}"/>
#             <arg value="-v"/>
#             <arg value="${release_dir}/${BEAST_package_name} v${version}.dmg"/>
#         </exec>
# codesign -s "Marc Suchard (LC992Z5R7E)" -v release/BEAST\ X\ v10.5.0.dmg
endif(APPLE)
