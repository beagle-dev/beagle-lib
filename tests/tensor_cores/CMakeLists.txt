include_directories(
        ${CUDA_INCLUDE_DIRS}
)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 80) #TODO: Make this generalizable?
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

foreach(STATE_SIZE IN LISTS STATE_SIZES)
    set(PRE_ORDER_TEST_NAME "test_pre_order_${STATE_SIZE}")
    add_executable(${PRE_ORDER_TEST_NAME} test_x_state_pre_order.cu)
    target_compile_definitions( ${PRE_ORDER_TEST_NAME} PRIVATE STATE_SIZE=${STATE_SIZE} )
    target_compile_options(${PRE_ORDER_TEST_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            -I/usr/local/cuda/include;
            -lineinfo
            >
    )
    target_link_libraries(${PRE_ORDER_TEST_NAME} cuda)
    set_source_files_properties(test_x_state_pre_order.cu PROPERTIES LANGUAGE CUDA)

    add_dependencies(${PRE_ORDER_TEST_NAME} TensorCoresKernels)
endforeach()

set(STATE_SIZE 32)

set(GRADIENT_TEST_NAME "test_gradient_${STATE_SIZE}")
add_executable(${GRADIENT_TEST_NAME} test_x_state_gradient.cu)
target_compile_definitions( ${GRADIENT_TEST_NAME} PRIVATE STATE_SIZE=${STATE_SIZE} )
target_compile_options(${GRADIENT_TEST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -I/usr/local/cuda/include;
        -lineinfo
        >
)
target_link_libraries(${GRADIENT_TEST_NAME} cuda)
set_source_files_properties(test_x_state_gradient.cu PROPERTIES LANGUAGE CUDA)

add_dependencies(${GRADIENT_TEST_NAME} TensorCoresKernels)