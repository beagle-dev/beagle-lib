
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(NVCC "/usr/local/cuda/bin/nvcc")
set(NVCCFLAGS "-O3 -arch compute_30 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE")
set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")

# For CUDA, we need to generate the file `BeagleCUDA_kernels.h` using the commands (and dependencies) below
add_custom_target(CudaKernels
        ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_cuda_kernels.sh ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building kernels")

add_library(hmsbeagle-cuda.${BEAGLE_VERSION} MODULE
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

add_dependencies(hmsbeagle-cuda.${BEAGLE_VERSION} CudaKernels)

target_link_libraries(hmsbeagle-cuda.${BEAGLE_VERSION} cuda ${CUDA_LIBRARIES})

add_library(hmsbeagle-cuda MODULE
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

install(TARGETS hmsbeagle-cuda.${BEAGLE_VERSION} hmsbeagle-cuda
        DESTINATION lib
        )
