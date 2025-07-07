if(BUILD_OPENMP)

	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		# Do we need to do anything?  Probably need to statically link omp to cpu *.so
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread -fopenmp")
	else(OpenMP_CXX_FOUND)
		message (FATAL_ERROR "No OpenMP Found")
	endif(OpenMP_CXX_FOUND)

	add_library(hmsbeagle-cpu-omp SHARED
        ../BeagleCPU4StateSSEImpl.h
        ../BeagleCPU4StateSSEImpl.hpp
        ../BeagleCPU4StateImpl.h
        ../BeagleCPU4StateImpl.hpp
        ../BeagleCPUSSEImpl.h
        ../BeagleCPUSSEImpl.hpp
        ../BeagleCPUImpl.h
        ../BeagleCPUImpl.hpp
        ../BeagleCPUOpenMPPlugin.cpp
        ../BeagleCPUOpenMPPlugin.h
        ../EigenDecomposition.h
        ../EigenDecompositionCube.h
        ../EigenDecompositionCube.hpp
        ../EigenDecompositionSquare.h
        ../EigenDecompositionSquare.hpp
        ../Precision.h
        ../SSEDefinitions.h
        )

	install(TARGETS hmsbeagle-cpu-omp
		DESTINATION ${BEAGLE_INSTALL_DIR}
		COMPONENT cpu_omp
	    )

	SET_TARGET_PROPERTIES(hmsbeagle-cpu-omp
    	PROPERTIES
	    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
		SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    	)
    	
endif(BUILD_OPENMP)
