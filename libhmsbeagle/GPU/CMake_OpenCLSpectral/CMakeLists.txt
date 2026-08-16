
include_directories(
        ${OpenCL_INCLUDE_DIRS}
)

add_definitions(-DFW_OPENCL)

if(WIN32)
    set(CMD_NAME createOpenCLSpectralHeader.bat)
else(WIN32)
    set(CMD_NAME make_opencl_spectral_kernels.sh)
endif(WIN32)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCLSpectral_kernels.h
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/${CMD_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../kernels
        DEPENDS
            "../kernels/kernels4.cu"
            "../kernels/kernels4Derivatives.cu"
            "../kernels/kernelsX.cu"
            "../kernels/kernelsXDerivatives.cu"
            "../kernels/kernelsAll.cu"
            "../kernels/kernelsSpectralIfDef.cu"
            "../AdjointBlockSize.h"
            "../kernels/kernelsSpectralIfDef4.cu"
        COMMENT "Building OpenCL spectral kernels"
        VERBATIM)

add_custom_target(OpenCLSpectralKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCLSpectral_kernels.h)

add_library(hmsbeagle-opencl-spectral SHARED
        ../BeagleGPUImpl.h
        ../BeagleGPUImpl.hpp
        ../BeagleGPUSpectralImpl.h
        ../BeagleGPUSpectralImpl.hpp
        ../GPUImplDefs.h
        ../GPUImplHelper.cpp
        ../GPUImplHelper.h
        ../GPUInterfaceOpenCL.cpp
        ../KernelLauncher.cpp
        ../KernelLauncher.h
        ../KernelResource.cpp
        ../KernelResource.h
        ../OpenCLSpectralPlugin.cpp
        ../OpenCLSpectralPlugin.h
        ../Precision.h
        )

add_dependencies(hmsbeagle-opencl-spectral OpenCLSpectralKernels)

target_compile_definitions(hmsbeagle-opencl-spectral PRIVATE BEAGLE_SPECTRAL_KERNELS)

target_link_libraries(hmsbeagle-opencl-spectral ${OpenCL_LIBRARIES})

install(TARGETS hmsbeagle-opencl-spectral
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT opencl_spectral
    )

SET_TARGET_PROPERTIES(hmsbeagle-opencl-spectral
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

target_compile_options(hmsbeagle-opencl-spectral PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
