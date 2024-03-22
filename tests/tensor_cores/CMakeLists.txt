include_directories(
        ${CUDA_INCLUDE_DIRS}
)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
set(NVCCFLAGS "-lineinfo") # TODO: Make -lineinfo a flag

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "native")
enable_language("CUDA")

set(STATE_SIZES 16 32 64)

foreach(STATE_SIZE IN LISTS STATE_SIZES)
    set(POST_ORDER_TEST_NAME "test_post_order_${STATE_SIZE}")
    add_executable(${POST_ORDER_TEST_NAME} test_x_state_post_order.cu)
    target_compile_definitions( ${POST_ORDER_TEST_NAME} PRIVATE STATE_SIZE=${STATE_SIZE} )
    target_compile_options(${POST_ORDER_TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            -I/usr/local/cuda/include;
            -lineinfo
            >
    )
    target_link_libraries(${POST_ORDER_TEST_NAME} cuda)
    set_source_files_properties(test_x_state_post_order.cu PROPERTIES LANGUAGE CUDA)

    add_dependencies(${POST_ORDER_TEST_NAME} TensorCoresKernels)
endforeach()