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