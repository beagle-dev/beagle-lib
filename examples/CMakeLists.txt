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

    # every GPU spectral kernel against CPU spectral; skipped (77) without a GPU resource that offers spectral
    add_executable(gpuspectraltest
            spectraltest/gpuspectraltest.cpp)
    target_link_libraries(gpuspectraltest
            hmsbeagle
            hmsbeagle-cpu
            hmsbeagle-cpu-spectral
            ${CMAKE_DL_LIBS})
    add_test(gpuspectraltest gpuspectraltest)
    set_tests_properties(gpuspectraltest PROPERTIES SKIP_RETURN_CODE 77)

    # spectral adjoint gradients against finite differences on the CPU (17 and 56 states), and on a GPU against the CPU
    target_link_libraries(adjointtest4 hmsbeagle-cpu-spectral)
    add_test(adjointtest4 adjointtest4 --all)
    if(BUILD_OPENCL OR BUILD_CUDA)
        add_test(adjointtest4-gpu adjointtest4 --gpu 1 --all)
        set_tests_properties(adjointtest4-gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif(BUILD_OPENCL OR BUILD_CUDA)
    if(BUILD_OPENCL)
        # the model at eigen index 1, so the adjoint reaches the eigen-indexed offsets of its pools; a
        # BEAGLE_DEBUG_MEMORY build also pads every sub-buffer, as AMD and Intel OpenCL do
        add_test(adjointtest4-gpu-eigenindex adjointtest4 --gpu 1 --all --eigenindex 1)
        set_tests_properties(adjointtest4-gpu-eigenindex PROPERTIES SKIP_RETURN_CODE 77)
        if(BEAGLE_DEBUG_MEMORY)
            set_tests_properties(adjointtest4-gpu-eigenindex PROPERTIES ENVIRONMENT BEAGLE_DEBUG_OPENCL_ALIGN=128)
        endif(BEAGLE_DEBUG_MEMORY)
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
    if(BUILD_OPENCL OR BUILD_CUDA)
        add_test(gpuspectraldensetest gpuspectraldensetest)
        set_tests_properties(gpuspectraldensetest PROPERTIES SKIP_RETURN_CODE 77)
    endif(BUILD_OPENCL OR BUILD_CUDA)

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
    if(BUILD_OPENCL OR BUILD_CUDA)
        # its GPU axis on resource 1, skipped without a GPU; a BEAGLE_DEBUG_MEMORY build also checks the counters and
        # forces allocations to fail, with every sub-buffer padded, as AMD and Intel OpenCL do
        add_test(buffergrowthtest-gpu buffergrowthtest --gpu 1)
        set_tests_properties(buffergrowthtest-gpu PROPERTIES SKIP_RETURN_CODE 77)
        if(BEAGLE_DEBUG_MEMORY)
            set_tests_properties(buffergrowthtest-gpu PROPERTIES ENVIRONMENT BEAGLE_DEBUG_OPENCL_ALIGN=128)
        endif(BEAGLE_DEBUG_MEMORY)
    endif(BUILD_OPENCL OR BUILD_CUDA)
endif(BUILD_SSE AND BUILD_SPECTRAL)

if(BUILD_TINYGPU)
    add_executable(tinygputest
            tinygputest/tinygputest.cpp)
    target_link_libraries(tinygputest
            hmsbeagle
            hmsbeagle-cpu
            ${CMAKE_DL_LIBS})
    if(BUILD_SSE)
        target_link_libraries(tinygputest hmsbeagle-cpu-sse)
    endif()
    # boots the attached eGPU through TinyGPU.app; with none it finds no TinyGPU resource and skips. The plugin is loaded by
    # name at run time, so ctest gives the test the build-tree plugin directory, where the crash guard is built too
    add_test(tinygputest tinygputest)
    get_target_property(TINYGPU_PLUGIN_DIR hmsbeagle-tinygpu BINARY_DIR)
    set_tests_properties(tinygputest PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "DYLD_LIBRARY_PATH=${TINYGPU_PLUGIN_DIR}")
endif()

if(BUILD_FUSED)
    # the fused updatePartials TEST implementation (BEAGLE_FLAG_PARALLELOPS_FUSED) against the standard GPU one in one
    # process; see fusedbench/fusedbench.cpp. fusedcheck skips (77) without a fused plugin. The plugins are loaded by
    # name at run time, so ctest gives it the build-tree directories of whichever OpenCL plugins exist
    add_executable(fusedbench
            fusedbench/fusedbench.cpp)
    target_link_libraries(fusedbench
            hmsbeagle
            hmsbeagle-cpu
            ${CMAKE_DL_LIBS})
    add_test(fusedcheck fusedbench --check --quick)
    set_tests_properties(fusedcheck PROPERTIES SKIP_RETURN_CODE 77)
    set(FUSED_PLUGIN_DIRS "")
    foreach(GPU_PLUGIN hmsbeagle-opencl hmsbeagle-opencl-spectral hmsbeagle-opencl-fused)
        if(TARGET ${GPU_PLUGIN})
            get_target_property(GPU_PLUGIN_DIR ${GPU_PLUGIN} BINARY_DIR)
            list(APPEND FUSED_PLUGIN_DIRS ${GPU_PLUGIN_DIR})
        endif()
    endforeach()
    if(FUSED_PLUGIN_DIRS)
        string(REPLACE ";" ":" FUSED_PLUGIN_DIRS "${FUSED_PLUGIN_DIRS}")
        set_tests_properties(fusedcheck PROPERTIES
            ENVIRONMENT "DYLD_LIBRARY_PATH=${FUSED_PLUGIN_DIRS};LD_LIBRARY_PATH=${FUSED_PLUGIN_DIRS}")
    endif()
endif(BUILD_FUSED)

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
	foreach(GPU_TEST gpumatrixtest adjointtest4-gpu-eigenindex hmctest5-gpu openclspectraltest)
		if(TEST ${GPU_TEST})
			set_property(TEST ${GPU_TEST} APPEND PROPERTY ENVIRONMENT
				"DYLD_LIBRARY_PATH=${GPU_PLUGIN_DIRS}" "LD_LIBRARY_PATH=${GPU_PLUGIN_DIRS}")
		endif()
	endforeach()
endif()
# These tests get the CUDA plugin directories as well, so that their GPU cases do not skip on a CUDA build. The CUDA
# plugins are loaded first (beagle.cpp), so with both frameworks built their GPU resource is a CUDA device
set(ALL_GPU_PLUGIN_DIRS "")
foreach(GPU_PLUGIN hmsbeagle-opencl hmsbeagle-opencl-spectral hmsbeagle-cuda hmsbeagle-cuda-spectral)
	if(TARGET ${GPU_PLUGIN})
		get_target_property(GPU_PLUGIN_DIR ${GPU_PLUGIN} BINARY_DIR)
		list(APPEND ALL_GPU_PLUGIN_DIRS ${GPU_PLUGIN_DIR})
	endif()
endforeach()
if(ALL_GPU_PLUGIN_DIRS)
	string(REPLACE ";" ":" ALL_GPU_PLUGIN_DIRS "${ALL_GPU_PLUGIN_DIRS}")
	foreach(GPU_TEST buffergrowthtest-gpu gpuspectraltest gpuspectraldensetest adjointtest4-gpu)
		if(TEST ${GPU_TEST})
			set_property(TEST ${GPU_TEST} APPEND PROPERTY ENVIRONMENT
				"DYLD_LIBRARY_PATH=${ALL_GPU_PLUGIN_DIRS}" "LD_LIBRARY_PATH=${ALL_GPU_PLUGIN_DIRS}")
		endif()
	endforeach()
endif()
