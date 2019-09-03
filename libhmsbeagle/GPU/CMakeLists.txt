
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
    add_subdirectory("CMake_OpenCL")
endif(OpenCL_FOUND)

#find_package(CUDA)
#if(CUDA_FOUND)
#    message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
#    add_subdirectory("CMake_CUDA")
#endif(CUDA_FOUND)
