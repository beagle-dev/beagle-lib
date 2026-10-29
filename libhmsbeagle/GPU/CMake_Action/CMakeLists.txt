
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

if(WIN32)
    set(NVCC "nvcc.exe")
    set(NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets")
    set(NVCC_INC "-I../../..")
    set(CMD_NAME createCUDAKernels.bat)
else(WIN32)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    set(NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE -std=c++11")
    set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")
    set(CMD_NAME make_cuda_kernels.sh)
#    set(CMAKE_CUDA_COMPILER ${NVCC})
endif(WIN32)

#set(CMAKE_CUDA_ARCHITECTURES "native")
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Wno-deprecated-gpu-targets --extended-lambda -std=c++17")



add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/${CMD_NAME} ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building CUDA kernels"
        VERBATIM)

#add_custom_target(CudaKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h)

add_library(hmsbeagle-action-cuda SHARED
        ../BeagleGPUActionImpl.h
        ../BeagleGPUActionImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceCUDA.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../ActionCUDAPlugin.cpp
        ../ActionCUDAPlugin.h
        ../cuda_ops.cu
        ../Precision.h
)

add_dependencies(hmsbeagle-action-cuda CudaKernels)

target_link_libraries(hmsbeagle-action-cuda cuda ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} Eigen3::Eigen)

install(TARGETS hmsbeagle-action-cuda
        DESTINATION ${BEAGLE_INSTALL_DIR}
        COMPONENT action-cuda
)

SET_TARGET_PROPERTIES(hmsbeagle-action-cuda
        PROPERTIES
        SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
        SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
)
