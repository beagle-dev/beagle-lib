add_executable(hmctest
        hmctest/hmctest.cpp)

add_executable(hmctest5
		hmctest/hmctest5.cpp)

add_executable(adjointtest4
		hmctest/adjointtest4.cpp)

add_executable(adjointbench
		hmctest/adjointbench.cpp)

add_executable(prunebench
		hmctest/prunebench.cpp)

#add_executable(hmcGaptest
#    	hmctest/hmcGapTest.cpp)

#add_executable(tinytest
#        tinytest/tinytest.cpp)

#add_executable(hmctestMedium
#        hmctest/hmctestMedium.cpp)

#add_executable(genomictest
#        genomictest/genomictest.cpp
#        genomictest/linalg.cpp
#        genomictest/linalg.h
#        )

add_executable(synthetictest
		synthetictest/synthetictest.cpp
		synthetictest/linalg.cpp
		synthetictest/linalg.h
		)

#add_executable(complextest
#        complextest/complextest.cpp)
if(BEAGLE_BENCHMARK)
#	list(APPEND BENCHMARK_LIBS nvidia-ml) # Needed for nvml.h
endif()

target_link_libraries(hmctest
	hmsbeagle
#	hmsbeagle-cpu-sse
	hmsbeagle-cpu
	${BENCHMARK_LIBS}
	${CMAKE_DL_LIBS})

if(BUILD_SSE)
	target_link_libraries(hmctest hmsbeagle-cpu-sse)
endif(BUILD_SSE)

target_link_libraries(hmctest5
	hmsbeagle
	hmsbeagle-cpu
	${BENCHMARK_LIBS}
	${CMAKE_DL_LIBS})
if(BUILD_SSE)
	target_link_libraries(hmctest5 hmsbeagle-cpu-sse)
endif(BUILD_SSE)

target_link_libraries(adjointtest4
	hmsbeagle
	hmsbeagle-cpu
	${BENCHMARK_LIBS}
	${CMAKE_DL_LIBS})

target_link_libraries(adjointbench
	hmsbeagle
	hmsbeagle-cpu
	${BENCHMARK_LIBS}
	${CMAKE_DL_LIBS})

target_link_libraries(prunebench
	hmsbeagle
	hmsbeagle-cpu
	${BENCHMARK_LIBS}
	${CMAKE_DL_LIBS})


target_link_libraries(synthetictest
		hmsbeagle
#		hmsbeagle-cpu-sse
		hmsbeagle-cpu
		${BENCHMARK_LIBS}
		${CMAKE_DL_LIBS})
if(BUILD_SSE)
	target_link_libraries(synthetictest hmsbeagle-cpu-sse)
endif(BUILD_SSE)

if(BUILD_SSE)
	target_link_libraries(hmctest
		hmsbeagle-cpu-sse)
		
	target_link_libraries(synthetictest
		hmsbeagle-cpu-sse)		
endif(BUILD_SSE)

add_test(hmctest hmctest)

# setTransitionMatrices, getTransitionMatrix, convolve and transpose on a GPU against the CPU; skipped without a GPU
add_executable(gpumatrixtest
		gpumatrixtest/gpumatrixtest.cpp)
target_link_libraries(gpumatrixtest
		hmsbeagle
		hmsbeagle-cpu
		${CMAKE_DL_LIBS})
if(BUILD_OPENCL)
	add_test(gpumatrixtest gpumatrixtest)
	set_tests_properties(gpumatrixtest PROPERTIES SKIP_RETURN_CODE 77)
endif(BUILD_OPENCL)

if(BUILD_SPECTRAL)
    add_executable(openclspectraltest
            openclspectraltest/openclspectraltest.cpp)
    target_link_libraries(openclspectraltest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(openclspectraltest openclspectraltest)

    add_executable(spectralcomplextest
            spectraltest/spectralcomplextest.cpp)
    target_link_libraries(spectralcomplextest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(spectralcomplextest spectralcomplextest)

    # every GPU spectral kernel against CPU spectral; skipped without a GPU
    add_executable(gpuspectraltest
            spectraltest/gpuspectraltest.cpp)
    target_link_libraries(gpuspectraltest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(gpuspectraltest gpuspectraltest)

    # spectral adjoint gradients against finite differences on the CPU (17 and 56 states), and on a GPU against the CPU
    target_link_libraries(adjointtest4 hmsbeagle-cpu-spectral)
    add_test(adjointtest4 adjointtest4 --all)
    if(BUILD_OPENCL)
        add_test(adjointtest4-gpu adjointtest4 --gpu 1 --all)
        set_tests_properties(adjointtest4-gpu PROPERTIES SKIP_RETURN_CODE 77)
        # pre-order with PREORDER_TRANSPOSE_AUTO and edge derivatives on a GPU; skipped without one
        add_test(hmctest5-gpu hmctest5 --gpu 1)
        set_tests_properties(hmctest5-gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif(BUILD_OPENCL)

    # dense matrices on GPU spectral (pool allocated by the first dense write) against CPU spectral; skipped without
    # a GPU
    add_executable(gpuspectraldensetest
            spectraltest/gpuspectraldensetest.cpp)
    target_link_libraries(gpuspectraldensetest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    if(BUILD_OPENCL)
        add_test(gpuspectraldensetest gpuspectraldensetest)
        set_tests_properties(gpuspectraldensetest PROPERTIES SKIP_RETURN_CODE 77)
    endif(BUILD_OPENCL)

    # standard vs spectral timings; see spectraltest/spectral_benchmark.py
    add_executable(spectralbench
            spectraltest/spectralbench.cpp)
    target_link_libraries(spectralbench
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    if(BUILD_SSE)
        target_link_libraries(spectralbench hmsbeagle-cpu-sse)
    endif(BUILD_SSE)
endif(BUILD_SPECTRAL)

if(BUILD_HIERARCHICAL)
    add_executable(hratetest
            hratetest/hratetest.cpp)
    target_link_libraries(hratetest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-hierarchical
            ${CMAKE_DL_LIBS})
    add_test(hratetest hratetest)
    add_test(hratetest-double hratetest --double)
    add_test(hratetest-states hratetest --states)
    add_test(hratetest-threads hratetest --threads 8)
    add_test(hratetest-scaling hratetest --scaling manual)
    add_test(hratetest-highstates hratetest --highstates)
    add_test(hratetest-kappa hratetest --kappa 2 --double)
    add_executable(hratelayouttest
            hratetest/hratelayouttest.cpp)
    target_link_libraries(hratelayouttest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-hierarchical
            ${CMAKE_DL_LIBS})
    add_test(hratelayouttest hratelayouttest)
    add_executable(hratebundletest
            hratetest/hratebundletest.cpp)
    target_link_libraries(hratebundletest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-hierarchical
            ${CMAKE_DL_LIBS})
    add_test(hratebundletest hratebundletest)
    add_executable(hratepruningtest
            hratetest/hratepruningtest.cpp)
    target_link_libraries(hratepruningtest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-hierarchical
            ${CMAKE_DL_LIBS})
    add_test(hratepruningtest hratepruningtest)
    foreach(HRATE_KERNEL tiled dense fused)
        add_test(hratepruningtest-${HRATE_KERNEL} hratepruningtest)
        set_property(TEST hratepruningtest-${HRATE_KERNEL} PROPERTY ENVIRONMENT "BEAGLE_HRATE_KERNEL=${HRATE_KERNEL}")
        add_test(hratebundletest-${HRATE_KERNEL} hratebundletest)
        set_property(TEST hratebundletest-${HRATE_KERNEL} PROPERTY ENVIRONMENT "BEAGLE_HRATE_KERNEL=${HRATE_KERNEL}")
        add_test(hratetest-states-${HRATE_KERNEL} hratetest --states)
        set_property(TEST hratetest-states-${HRATE_KERNEL} PROPERTY ENVIRONMENT "BEAGLE_HRATE_KERNEL=${HRATE_KERNEL}")
    endforeach()
    foreach(HRATE_BUDGET 0 2000)
        add_test(hratepruningtest-fused-budget-${HRATE_BUDGET} hratepruningtest)
        set_property(TEST hratepruningtest-fused-budget-${HRATE_BUDGET} PROPERTY ENVIRONMENT
                     "BEAGLE_HRATE_KERNEL=fused;BEAGLE_HRATE_TABLE_BUDGET=${HRATE_BUDGET}")
    endforeach()
    add_executable(hratebench
            hratetest/hratebench.cpp)
    target_link_libraries(hratebench
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-hierarchical
            ${CMAKE_DL_LIBS})
    if(BUILD_SSE)
        target_link_libraries(hratebench hmsbeagle-cpu-sse)
    endif(BUILD_SSE)
    if(APPLE)
        target_link_libraries(hratebench "-framework Accelerate")
        target_compile_definitions(hratebench PRIVATE ACCELERATE_NEW_LAPACK)
    endif()
    add_test(hratebench-smoke hratebench --states 17 --patterns 1008 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-smoke-standard hratebench --impl standard --vector none --states 17 --patterns 1008 --tips 8 --budget 0.2 --minreps 2)
    add_test(hratebench-single-vs-double-4 hratebench --precision single --states 4 --patterns 1008 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-single-vs-double-17 hratebench --precision single --states 17 --patterns 1008 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-fused-smoke hratebench --kernel fused --states 17 --patterns 1008 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-fused-states hratebench --kernel fused --tipdata states --states 20 --patterns 1003 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-fused-scaling hratebench --kernel fused --scaling always --states 20 --patterns 1003 --tips 8 --budget 0.2 --minreps 2 --check)
    add_test(hratebench-fused-single-4 hratebench --kernel fused --precision single --states 4 --patterns 1008 --tips 8 --budget 0.2 --minreps 2 --check)
endif(BUILD_HIERARCHICAL)

if(BUILD_SSE)
    # SIMD CPU implementations against their scalar counterparts
    add_executable(simdtest
            simdtest/simdtest.cpp)
    target_link_libraries(simdtest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-sse
            ${CMAKE_DL_LIBS})
    if(BUILD_SPECTRAL)
        target_link_libraries(simdtest hmsbeagle-cpu-spectral)
    endif(BUILD_SPECTRAL)
    add_test(simdtest simdtest)
endif(BUILD_SSE)

if(BUILD_SSE AND BUILD_SPECTRAL)
    # spectral dense matrix storage (written on demand) against the standard SSE implementation
    add_executable(spectraldensetest
            spectraltest/spectraldensetest.cpp)
    target_link_libraries(spectraldensetest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-sse
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(spectraldensetest spectraldensetest)

    # BEAGLE_INDEX_CHECKS rejects out-of-range buffer indices; without the option the test exits 77 (skipped)
    add_executable(indexchecktest
            indexchecktest/indexchecktest.cpp)
    target_link_libraries(indexchecktest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-sse
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(indexchecktest indexchecktest)
    set_tests_properties(indexchecktest PROPERTIES SKIP_RETURN_CODE 77)

    # beagleEnsureBufferCounts: instances grown in steps against instances created at the final counts
    add_executable(buffergrowthtest
            buffergrowthtest/buffergrowthtest.cpp)
    target_link_libraries(buffergrowthtest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-sse
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(buffergrowthtest buffergrowthtest)
endif(BUILD_SSE AND BUILD_SPECTRAL)

add_executable(hellobeagle
		standalone/hellobeagle/hello.cpp)
target_link_libraries(hellobeagle hmsbeagle)

#target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(hmcGaptest hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)

# The GPU plugins are loaded by name at run time and no example links them, so ctest gives the GPU tests the
# build-tree plugin directories (otherwise they find no GPU resource and skip)
if(TARGET hmsbeagle-opencl)
	get_target_property(GPU_PLUGIN_DIRS hmsbeagle-opencl BINARY_DIR) # where the plugin is built
	if(TARGET hmsbeagle-opencl-spectral)
		get_target_property(GPU_SPECTRAL_PLUGIN_DIR hmsbeagle-opencl-spectral BINARY_DIR)
		string(APPEND GPU_PLUGIN_DIRS ":${GPU_SPECTRAL_PLUGIN_DIR}")
	endif()
	foreach(GPU_TEST gpumatrixtest adjointtest4-gpu hmctest5-gpu gpuspectraltest openclspectraltest gpuspectraldensetest)
		if(TEST ${GPU_TEST})
			set_property(TEST ${GPU_TEST} APPEND PROPERTY ENVIRONMENT
				"DYLD_LIBRARY_PATH=${GPU_PLUGIN_DIRS}" "LD_LIBRARY_PATH=${GPU_PLUGIN_DIRS}")
		endif()
	endforeach()
endif()
