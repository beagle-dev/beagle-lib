if(BUILD_OPENMP)

	option(BUILD_OPENMP_WITH_HOMEBREW "Build beagle with OpenMP library" OFF)

	if(BUILD_OPENMP_WITH_HOMEBREW)
	    set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/opt/homebrew/opt/llvm/include")
   		set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/opt/homebrew/opt/llvm/include")
    
	    set(OpenMP_C_LIB_NAMES omp)   
    	set(OpenMP_CXX_LIB_NAMES omp)
    
	    set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
	    set(OpenMP_C_INCLUDE_DIR "/opt/homebrew/opt/llvm/include")
	endif(BUILD_OPENMP_WITH_HOMEBREW)
    
    find_package(OpenMP REQUIRED)

 	if(OpenMP_CXX_FOUND)
 		message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
 		message(STATUS "OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}")

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

	# Apply OpenMP compilation flags and link OpenMP libraries
	target_compile_options(hmsbeagle-cpu-omp PRIVATE ${OpenMP_CXX_FLAGS})
	target_link_libraries(hmsbeagle-cpu-omp PRIVATE OpenMP::OpenMP_CXX)
	
	# Add preprocessor definition to enable OpenMP threading flags in CPU implementations
	target_compile_definitions(hmsbeagle-cpu-omp PRIVATE BEAGLE_OPENMP_ENABLED)

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
