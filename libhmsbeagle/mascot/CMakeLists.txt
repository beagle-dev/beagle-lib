# MASCOT component: a C API (mascot.h) over two runtime-K kernels (A and B), each built strict and with FMA.
# Self-contained: no BEAGLE headers or buffers, so it merges into master and matrix_vector alike.

# The kernels are C++17 and are tuned at -O3; RelWithDebInfo's -O2 would otherwise come after the global -O3.
set(MASCOT_COMMON_OPTIONS)
if(MSVC)
    list(APPEND MASCOT_COMMON_OPTIONS /O2 /fp:precise)
else()
    list(APPEND MASCOT_COMMON_OPTIONS -O3)
endif()

# Strict: no contraction at all, so results are bitwise those of the Java reference.
set(MASCOT_STRICT_OPTIONS ${MASCOT_COMMON_OPTIONS})
if(NOT MSVC)
    # -fno-math-errno: the kernels never read errno, and under -fmath-errno std::fma is a libm call
    list(APPEND MASCOT_STRICT_OPTIONS -ffp-contract=off -fno-math-errno)
endif()

# FMA: also no automatic contraction; the kernels fuse exactly where they write MASCOT_MUL_ADD (MascotFma.hpp).
# The x86 code needs the fma target and is only chosen at run time when the CPU reports FMA.
set(MASCOT_FMA_OPTIONS ${MASCOT_STRICT_OPTIONS})
if(NOT MSVC)
    if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        list(APPEND MASCOT_FMA_OPTIONS "SHELL:-Xarch_x86_64 -mfma")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
        list(APPEND MASCOT_FMA_OPTIONS -mfma)
    endif()
endif()

foreach(kernel A B)
    foreach(mode strict fma)
        set(target mascot_kernel_${kernel}_${mode})
        add_library(${target} OBJECT MascotKernel${kernel}.cpp)
        set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                              POSITION_INDEPENDENT_CODE ON)
        if(mode STREQUAL "fma")
            target_compile_definitions(${target} PRIVATE MASCOT_FP_FMA)
            target_compile_options(${target} PRIVATE ${MASCOT_FMA_OPTIONS})
        else()
            target_compile_options(${target} PRIVATE ${MASCOT_STRICT_OPTIONS})
        endif()
    endforeach()
endforeach()

add_library(hmsbeagle-mascot STATIC
        mascot.cpp
        mascot.h
        MascotProblem.hpp
        $<TARGET_OBJECTS:mascot_kernel_A_strict>
        $<TARGET_OBJECTS:mascot_kernel_A_fma>
        $<TARGET_OBJECTS:mascot_kernel_B_strict>
        $<TARGET_OBJECTS:mascot_kernel_B_fma>
        )
set_target_properties(hmsbeagle-mascot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                      POSITION_INDEPENDENT_CODE ON)
target_compile_options(hmsbeagle-mascot PRIVATE ${MASCOT_STRICT_OPTIONS})
if(NOT WIN32)
    target_link_libraries(hmsbeagle-mascot PUBLIC pthread)
endif()

# JNI library for BEAST (beagle.mascot.MascotJNIWrapper); BEAST loads it from java.library.path or mascot.library.path.
if(BUILD_JNI AND JNI_FOUND)
    add_library(hmsbeagle-jni-mascot SHARED JNI/beagle_mascot_MascotJNIWrapper.cpp)
    set_target_properties(hmsbeagle-jni-mascot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_compile_options(hmsbeagle-jni-mascot PRIVATE ${MASCOT_STRICT_OPTIONS})
    target_include_directories(hmsbeagle-jni-mascot PRIVATE ${JNI_INCLUDE_DIRS})
    target_link_libraries(hmsbeagle-jni-mascot hmsbeagle-mascot)
    install(TARGETS hmsbeagle-jni-mascot DESTINATION ${BEAGLE_INSTALL_DIR})
endif()

# Tests run with "ctest -R mascot" (other tests in this tree may use GPUs).
# FMA tolerance, relative (see tests/MascotGolden.hpp): the explicit-FMA kernels differ from the strict reference by
# at most 1.9e-13 on the golden corpus (2026-10-30).
set(MASCOT_FMA_RTOL 1e-11)

foreach(test mascot_selftest mascot_golden_test mascot_api_test mascot_bench)
    add_executable(${test} tests/${test}.cpp)
    set_target_properties(${test} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_compile_options(${test} PRIVATE ${MASCOT_STRICT_OPTIONS})
    target_link_libraries(${test} hmsbeagle-mascot)
endforeach()

# Self-contained (CI): generated cases; kernels A and B must agree bit for bit in the strict build, FMA within the
# tolerance, gradients must match finite differences; also statuses, dispatch and concurrent instances.
add_test(NAME mascot_selftest COMMAND mascot_selftest --rtol ${MASCOT_FMA_RTOL})

# Golden tests against BEAST's Java reference (opt-in): point MASCOT_GOLDEN_DIR at a directory of *.golden cases
# written by BEAST's dr.evomodel.coalescent.mascot.MascotGoldenCorpus. mascot_bench times such cases (not a test).
set(MASCOT_GOLDEN_DIR "" CACHE PATH "Directory of MASCOT golden cases (*.golden) from BEAST; empty skips the golden tests")
if(MASCOT_GOLDEN_DIR)
    file(GLOB MASCOT_GOLDEN_CASES "${MASCOT_GOLDEN_DIR}/*.golden")
    list(SORT MASCOT_GOLDEN_CASES)
    if(NOT MASCOT_GOLDEN_CASES)
        message(FATAL_ERROR "MASCOT_GOLDEN_DIR=${MASCOT_GOLDEN_DIR} contains no *.golden cases")
    endif()
    foreach(kernel A B)
        add_test(NAME mascot_golden_${kernel}_strict
                 COMMAND mascot_golden_test --kernel ${kernel} ${MASCOT_GOLDEN_CASES})
        add_test(NAME mascot_golden_${kernel}_fma
                 COMMAND mascot_golden_test --kernel ${kernel} --fma --rtol ${MASCOT_FMA_RTOL} ${MASCOT_GOLDEN_CASES})
    endforeach()
    add_test(NAME mascot_api_strict COMMAND mascot_api_test ${MASCOT_GOLDEN_CASES})
    add_test(NAME mascot_api_fma COMMAND mascot_api_test --fma --rtol ${MASCOT_FMA_RTOL} ${MASCOT_GOLDEN_CASES})
endif()
