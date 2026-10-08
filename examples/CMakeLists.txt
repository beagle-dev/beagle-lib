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
endif(BUILD_SSE AND BUILD_SPECTRAL)

add_executable(hellobeagle
		standalone/hellobeagle/hello.cpp)
target_link_libraries(hellobeagle hmsbeagle)

#target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(hmcGaptest hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)
