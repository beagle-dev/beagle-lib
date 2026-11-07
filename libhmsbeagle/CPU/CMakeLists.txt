# PROTOTYPE batched adjoint (BeagleCPUAdjointBatched.hpp, in every CPU implementation): GEMMs from Accelerate on
# macOS, otherwise a plain loop
function(beagle_cpu_blas target)
	if(APPLE)
		target_link_libraries(${target} "-framework Accelerate")
		target_compile_definitions(${target} PRIVATE BEAGLE_HAVE_ACCELERATE)
	endif(APPLE)
endfunction()

add_library(hmsbeagle-cpu SHARED
        BeagleCPU4StateImpl.h
        BeagleCPU4StateImpl.hpp
        BeagleCPUImpl.h
        BeagleCPUImpl.hpp
        BeagleCPUPlugin.cpp
        BeagleCPUPlugin.h
        EigenDecomposition.h
        EigenDecompositionCube.h
        EigenDecompositionCube.hpp
        EigenDecompositionSquare.h
        EigenDecompositionSquare.hpp
        Precision.h
        SSEDefinitions.h
        BeagleCPUAdjointBatched.hpp
        )
beagle_cpu_blas(hmsbeagle-cpu)

install(TARGETS hmsbeagle-cpu
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT cpu
    )

SET_TARGET_PROPERTIES(hmsbeagle-cpu
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

if(NOT MSVC)
	target_compile_options(hmsbeagle-cpu PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif(NOT MSVC)

if(BUILD_SSE)
add_library(hmsbeagle-cpu-sse SHARED
        BeagleCPU4StateSSEImpl.h
        BeagleCPU4StateSSEImpl.hpp
        BeagleCPU4StateImpl.h
        BeagleCPU4StateImpl.hpp
        BeagleCPUSSEImpl.h
        BeagleCPUSSEImpl.hpp
        BeagleCPUImpl.h
        BeagleCPUImpl.hpp
        BeagleCPUSSEPlugin.cpp
        BeagleCPUSSEPlugin.h
        EigenDecomposition.h
        EigenDecompositionCube.h
        EigenDecompositionCube.hpp
        EigenDecompositionSquare.h
        EigenDecompositionSquare.hpp
        Precision.h
        SSEDefinitions.h
        SSEKernels.h
        BeagleCPUAdjointSSE.h
        BeagleCPUAdjointSSE.hpp
        BeagleCPUAdjointBatched.hpp
        BeagleCPUAdjoint4StateSSE.h
        BeagleCPUAdjoint4StateSSE.hpp
        )

beagle_cpu_blas(hmsbeagle-cpu-sse)

install(TARGETS hmsbeagle-cpu-sse
	DESTINATION ${BEAGLE_INSTALL_DIR}
	COMPONENT cpu_sse
    )

SET_TARGET_PROPERTIES(hmsbeagle-cpu-sse
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

if(NOT MSVC)
	target_compile_options(hmsbeagle-cpu-sse PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif(NOT MSVC)
endif(BUILD_SSE)

if(BUILD_SPECTRAL)
add_library(hmsbeagle-cpu-spectral SHARED
        BeagleCPUSpectralImpl.h
        BeagleCPUSpectralImpl.hpp
        BeagleCPUSpectralPlugin.cpp
        BeagleCPUSpectralPlugin.h
        BeagleCPUSpectralSSEImpl.h
        BeagleCPUSpectralSSEImpl.hpp
        BeagleCPU4StateSpectralSSEImpl.h
        BeagleCPU4StateSpectralSSEImpl.hpp
        SSEDefinitions.h
        SSEKernels.h
        BeagleCPUAdjointSSE.h
        BeagleCPUAdjointSSE.hpp
        BeagleCPUAdjoint4StateSSE.h
        BeagleCPUAdjoint4StateSSE.hpp
        EigenDecomposition.h
        EigenDecompositionSpectral.h
        EigenDecompositionSpectral.hpp
        AdjointMethods.h
        AdjointMethods.hpp
        BeagleCPUAdjointBatched.hpp
        Precision.h
        )

beagle_cpu_blas(hmsbeagle-cpu-spectral)
        
# target_compile_options(hmsbeagle-cpu-spectral PRIVATE
#     -fno-math-errno      # safe: no errno for math funcs
#     -fno-trapping-math   # safe: assumes FP exceptions don't trap
#     )        
#     
# target_compile_options(hmsbeagle-cpu-spectral PRIVATE
#     -Rpass=loop-vectorize          # shows vectorized loops
#     -Rpass-missed=loop-vectorize   # shows WHY a loop was skipped
#     -Rpass-analysis=loop-vectorize # detailed analysis
# 	)    

install(TARGETS hmsbeagle-cpu-spectral
	DESTINATION ${BEAGLE_INSTALL_DIR}
	COMPONENT cpu_spectral
    )

SET_TARGET_PROPERTIES(hmsbeagle-cpu-spectral
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

if(NOT MSVC)
	target_compile_options(hmsbeagle-cpu-spectral PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif(NOT MSVC)
endif(BUILD_SPECTRAL)

if(BUILD_OPENMP)
	add_subdirectory("CMake_OpenMP")
endif(BUILD_OPENMP)
