# hmsbeagle-opencl exports only plugin_init (-fvisibility=hidden, CMake_OpenCL/CMakeLists.txt), so the test compiles the
# GPUInterface it calls instead of linking the plugin
add_executable(test_opencl_discovery
    test_opencl_discovery.cpp
    ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp
    ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU/GPUImplHelper.cpp
    ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU/KernelResource.cpp)
add_dependencies(test_opencl_discovery OpenCLKernels)
target_compile_definitions(test_opencl_discovery PRIVATE FW_OPENCL)
target_include_directories(test_opencl_discovery PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(test_opencl_discovery PRIVATE ${OpenCL_LIBRARIES})
add_test(NAME opencl_discovery_handles_enumeration_error COMMAND test_opencl_discovery)

add_executable(test_opencl_cpu_instance
    test_opencl_cpu_instance.cpp)
target_link_libraries(test_opencl_cpu_instance PRIVATE hmsbeagle hmsbeagle-cpu hmsbeagle-opencl)
add_test(NAME opencl_plugin_allows_cpu_instance COMMAND test_opencl_cpu_instance)

if(APPLE)
    set(BEAGLE_PLUGIN_TEST_PATH
        "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:hmsbeagle>:$<TARGET_FILE_DIR:hmsbeagle-cpu>:$<TARGET_FILE_DIR:hmsbeagle-opencl>")
elseif(UNIX)
    set(BEAGLE_PLUGIN_TEST_PATH
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:hmsbeagle>:$<TARGET_FILE_DIR:hmsbeagle-cpu>:$<TARGET_FILE_DIR:hmsbeagle-opencl>")
endif()

if(BEAGLE_PLUGIN_TEST_PATH)
    set_tests_properties(opencl_plugin_allows_cpu_instance PROPERTIES
        ENVIRONMENT "${BEAGLE_PLUGIN_TEST_PATH}")
endif()
