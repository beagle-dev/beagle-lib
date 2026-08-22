
if(BUILD_OPENCL)
	find_package(OpenCL)
	if(OpenCL_FOUND)
		message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
		message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
		add_subdirectory("CMake_OpenCL")
		add_subdirectory("${PROJECT_SOURCE_DIR}/tests/opencl"
		                 "${PROJECT_BINARY_DIR}/tests/opencl")
	endif(OpenCL_FOUND)
endif(BUILD_OPENCL)

if(BUILD_CUDA)
	find_package(CUDA)
	if(CUDA_FOUND)
		message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
		add_subdirectory("CMake_CUDA")
	endif(CUDA_FOUND)
endif(BUILD_CUDA)

if(CUDA_FOUND AND BUILD_CUDA AND BEAGLE_TENSOR_CORES)
    message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
    add_subdirectory("CMake_TensorCores")
endif()

