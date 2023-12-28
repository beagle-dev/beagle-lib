
find_package(OpenCL)
if(BUILD_OPENCL)
#    set(OpenCL_INCLUDE_DIRS /opt/rocm/opencl/include)
#    set(OpenCL_LIBRARIES /opt/rocm/opencl/lib/libOpenCL.so)
    if(OpenCL_FOUND OR (OpenCL_INCLUDE_DIRS AND OpenCL_LIBRARIES))
        message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
        add_subdirectory("CMake_OpenCL")
    endif(OpenCL_FOUND OR (OpenCL_INCLUDE_DIRS AND OpenCL_LIBRARIES))
endif(BUILD_OPENCL)

find_package(CUDA)
if(CUDA_FOUND AND BUILD_CUDA)
    message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
    add_subdirectory("CMake_CUDA")
endif()

if(CUDA_FOUND AND BUILD_CUDA AND BEAGLE_TENSOR_CORES)
    message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
    add_subdirectory("CMake_TensorCores")
endif()
                                    
