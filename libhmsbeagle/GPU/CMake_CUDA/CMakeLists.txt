
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
     ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

if(WIN32)
    set(NVCC "nvcc.exe")
    set(NVCCFLAGS "-O3 -arch compute_30 -Wno-deprecated-gpu-targets")
    set(NVCC_INC "-I../../..")
    set(CMD_NAME createCUDAKernals.bat)
else(WIN32)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    set(NVCCFLAGS "-O3 -arch compute_30 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE -std=c++11")
    set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")
     set(CMD_NAME make_cuda_kernels.sh)
endif(WIN32)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/${CMD_NAME} ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building CUDA kernels"
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
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT cuda
    )

SET_TARGET_PROPERTIES(hmsbeagle-cuda
    PROPERTIES
    SOVERSION ${BEAGLE_PLUGIN_VERSION}
	SUFFIX ${BEAGLE_PLUGIN_SUFFIX}
    )
