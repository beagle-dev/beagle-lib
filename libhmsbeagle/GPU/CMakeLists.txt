
find_package(OpenCL)
if(OpenCL_FOUND AND BUILD_OPENCL)
    message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
    add_subdirectory("CMake_OpenCL")
endif(OpenCL_FOUND AND BUILD_OPENCL)

find_package(CUDA)
if(CUDA_FOUND AND BUILD_CUDA)
    message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
    add_subdirectory("CMake_CUDA")
endif(CUDA_FOUND AND BUILD_CUDA)
