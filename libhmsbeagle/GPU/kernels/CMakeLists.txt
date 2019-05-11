# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below

add_custom_target(OpenKernels
        ${CMAKE_CURRENT_SOURCE_DIR}/make_opencl_kernels.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building kernels")

#add_custom_command(OUTPUT BeagleOpenCL_kernels.h
#        COMMAND ./make_opencl_kernels.sh)