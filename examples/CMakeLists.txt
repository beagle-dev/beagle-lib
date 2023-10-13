add_executable(hmctest
        hmctest/hmctest.cpp)

add_executable(hmctest5
		hmctest/hmctest5.cpp)

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

#target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(hmcGaptest hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)
