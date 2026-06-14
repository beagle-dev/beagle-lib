
# ROCm/HIP backend — kernels compiled at runtime via HIPRTC.

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)
find_package(hiprtc QUIET)

message(STATUS "ROCm HIP found: ${hip_VERSION}")

include_directories(${hip_INCLUDE_DIRS} /opt/rocm/include)

add_definitions(-DFW_ROCM)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleROCm_kernels.h
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_rocm_kernels.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS
            "../kernels/kernels4.cu"
            "../kernels/kernels4CudaCore.cu"
            "../kernels/kernels4Derivatives.cu"
            "../kernels/kernels4DerivativesCudaCore.cu"
            "../kernels/kernelsX.cu"
            "../kernels/kernelsXCudaCore.cu"
            "../kernels/kernelsXDerivatives.cu"
            "../kernels/kernelsXDerivativesCudaCore.cu"
            "../kernels/kernelsAll.cu"
        COMMENT "Building ROCm kernel strings"
        VERBATIM)

add_custom_target(ROCmKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleROCm_kernels.h)

add_library(hmsbeagle-rocm SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceROCm.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../ROCmPlugin.cpp
        ../ROCmPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-rocm ROCmKernels)

target_link_libraries(hmsbeagle-rocm hip::host hiprtc)

install(TARGETS hmsbeagle-rocm
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT rocm
    )

SET_TARGET_PROPERTIES(hmsbeagle-rocm
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )
