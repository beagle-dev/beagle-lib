
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(NVCC "/usr/local/cuda/bin/nvcc")
set(NVCCFLAGS "-O3 -arch compute_30 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE")
set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")

# For CUDA, we need to generate the file `BeagleCUDA_kernels.h` using the commands (and dependencies) below
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_cuda_kernels.sh ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building kernels"
        VERBATIM)

add_custom_target(CudaKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h)

add_library(hmsbeagle-cuda SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceCUDA.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../CUDAPlugin.cpp
        ../CUDAPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-cuda CudaKernels)

target_link_libraries(hmsbeagle-cuda cuda ${CUDA_LIBRARIES})

install(TARGETS hmsbeagle-cuda
        DESTINATION lib
        )

SET_TARGET_PROPERTIES(hmsbeagle-cuda
        PROPERTIES
        SOVERSION ${BEAGLE_VERSION}
        SUFFIX .so
        )
