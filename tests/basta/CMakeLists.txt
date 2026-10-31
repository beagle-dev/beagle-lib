include_directories(
        ${CUDA_INCLUDE_DIRS}
)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

enable_language("CUDA")

set(STATE_SIZE 16)

set(BASTA_TEST_NAME "test_updateInnerBastaPartials_${STATE_SIZE}")
add_executable(${BASTA_TEST_NAME} test_x_state_updateInnerBastaPartials.cu)
target_compile_definitions( ${BASTA_TEST_NAME} PRIVATE STATE_SIZE=${STATE_SIZE} )
target_compile_options(${BASTA_TEST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -I/usr/local/cuda/include;
        -lineinfo
        >
)
target_link_libraries(${BASTA_TEST_NAME} cuda)
set_source_files_properties(test_x_state_updateInnerBastaPartials.cu PROPERTIES LANGUAGE CUDA)

add_dependencies(${BASTA_TEST_NAME} CudaKernels)