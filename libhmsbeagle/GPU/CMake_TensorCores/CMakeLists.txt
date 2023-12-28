
include_directories(
        ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DCUDA -DCUDA_TENSOR_CORES)

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
    set(NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE -std=c++11 -arch=sm_80 -lineinfo")
    set(NVCC_INC "-I${CMAKE_SOURCE_DIR}")
     set(CMD_NAME make_tensor_core_kernels.sh)
endif(WIN32)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleTensorCore_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/${CMD_NAME} ${NVCC} "${NVCCFLAGS}" "${NVCC_INC}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS "../kernels/kernels4.cu" "../kernels/kernels4TensorCore.cu" "../kernels/kernels4Derivatives.cu" "../kernels/kernels4DerivativesTensorCore.cu" "../kernels/kernelsX.cu" "../kernels/kernelsXTensorCore.cu" "../kernels/kernelsXDerivatives.cu" "../kernels/kernelsXDerivativesTensorCore.cu" "../kernels/kernelsAll.cu"
        COMMENT "Building Tensor Core kernels"
        VERBATIM)

add_custom_target(TensorCoresKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleTensorCore_kernels.h)

add_library(hmsbeagle-tensor-cores SHARED
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
        ../TensorCoresPlugin.cpp
        ../TensorCoresPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-tensor-cores TensorCoresKernels)

target_link_libraries(hmsbeagle-tensor-cores cuda ${CUDA_LIBRARIES})

install(TARGETS hmsbeagle-tensor-cores
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT cuda_tensor_cores
    )

SET_TARGET_PROPERTIES(hmsbeagle-tensor-cores
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )
