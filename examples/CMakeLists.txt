add_executable(hmctest
        hmctest/hmctest.cpp)

add_executable(hmctest5
		hmctest/hmctest5.cpp)

add_executable(actiontest
		actiontest/actiontest.cpp)

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

target_link_libraries(actiontest
		hmsbeagle
		hmsbeagle-cpu-sse
		hmsbeagle-cpu
		${CMAKE_DL_LIBS})

target_link_libraries(hmctest
	hmsbeagle
	hmsbeagle-cpu-sse
	hmsbeagle-cpu
	${CMAKE_DL_LIBS})

target_link_libraries(hmctest5
	hmsbeagle
	hmsbeagle-cpu-sse
	hmsbeagle-cpu
	${CMAKE_DL_LIBS})

target_link_libraries(synthetictest
		hmsbeagle
		hmsbeagle-cpu-sse
		hmsbeagle-cpu
		${CMAKE_DL_LIBS})

add_test(hmctest hmctest)

#target_link_libraries(hmctest5 hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(hmcGaptest hmsbeagle ${CMAKE_DL_LIBS})
#target_link_libraries(genomictest beagle_lib)
#target_link_libraries(tinytest hmsbeagle)
