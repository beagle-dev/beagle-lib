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
endif(BUILD_SSE)
