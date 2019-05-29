
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA -DDLS_MACOS)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
##add_custom_target(CudaKernels
##        ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_opencl_kernels.sh
##        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
##        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
##        COMMENT "Building kernels")


set(NVCC "/usr/local/cuda/bin/nvcc")
set(NVCCFLAGS "-O3 -arch compute_30 -Wno-deprecated-gpu-targets -ccbin /usr/bin/clang -m64 -D_POSIX_C_SOURCE")
set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")

# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
add_custom_target(CudaKernels
        ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/make_cuda_kernels.sh ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building kernels")

add_library(hmsbeagle-cuda.31 MODULE
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

add_dependencies(hmsbeagle-cuda.31 CudaKernels)

target_link_libraries(hmsbeagle-cuda.31 cuda ${CUDA_LIBRARIES})

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

install(TARGETS hmsbeagle-cuda.31 hmsbeagle-cuda
        DESTINATION lib
        )
