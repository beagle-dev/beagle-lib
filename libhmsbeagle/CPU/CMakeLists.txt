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
        )

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
        BeagleCPUAdjoint4StateSSE.h
        BeagleCPUAdjoint4StateSSE.hpp
        )

install(TARGETS hmsbeagle-cpu-sse
	DESTINATION ${BEAGLE_INSTALL_DIR}
	COMPONENT cpu_sse
    )

SET_TARGET_PROPERTIES(hmsbeagle-cpu-sse
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

# The plugin uses SSE4.1 intrinsics on x86 (on arm64, sse2neon.h provides them), which need -msse4.1 unless
# -march=native already enables them.  The plugin checks for SSE4.1 when it is loaded.
include(CheckCXXCompilerFlag)
if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
	# universal build: only the x86_64 slice
	target_compile_options(hmsbeagle-cpu-sse PRIVATE "SHELL:-Xarch_x86_64 -msse4.1")
elseif(NOT MSVC)
	# (compilers for arm64 reject the flag)
	CHECK_CXX_COMPILER_FLAG("-msse4.1" COMPILER_SUPPORTS_SSE41)
	if(COMPILER_SUPPORTS_SSE41)
		target_compile_options(hmsbeagle-cpu-sse PRIVATE -msse4.1)
	endif()
endif()


if(NOT MSVC)
	target_compile_options(hmsbeagle-cpu-sse PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif(NOT MSVC)
endif(BUILD_SSE)

if(BUILD_ACTION)
add_library(hmsbeagle-cpu-action SHARED
		BeagleCPUImpl.h
		BeagleCPUImpl.hpp
		BeagleCPUActionPlugin.cpp
		BeagleCPUActionPlugin.h
		BeagleCPUActionImpl.h
		Precision.h
		BeagleCPUActionImpl.hpp)
target_link_libraries(hmsbeagle-cpu-action Eigen3::Eigen)
set(CMAKE_CXX_STANDARD 17)

install(TARGETS hmsbeagle-cpu-action
		DESTINATION ${BEAGLE_INSTALL_DIR}
		COMPONENT cpu_action
		)

SET_TARGET_PROPERTIES(hmsbeagle-cpu-action
		PROPERTIES
		SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
		SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
		)
endif(BUILD_ACTION)

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
        Precision.h
        )
        
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

if(BUILD_HIERARCHICAL)
add_library(hmsbeagle-cpu-hierarchical SHARED
        BeagleCPUHierarchicalImpl.h
        BeagleCPUHierarchicalImpl.hpp
        BeagleCPUHierarchicalPlugin.cpp
        BeagleCPUHierarchicalPlugin.h
        HierarchicalMethods.h
        HierarchicalMethods.hpp
        EigenDecomposition.h
        Precision.h
        )

install(TARGETS hmsbeagle-cpu-hierarchical
	DESTINATION ${BEAGLE_INSTALL_DIR}
	COMPONENT cpu_hierarchical
    )

SET_TARGET_PROPERTIES(hmsbeagle-cpu-hierarchical
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
	SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

if(NOT MSVC)
	target_compile_options(hmsbeagle-cpu-hierarchical PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif(NOT MSVC)
endif(BUILD_HIERARCHICAL)
