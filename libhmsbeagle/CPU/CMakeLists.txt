add_library(hmsbeagle-cpu.30 MODULE
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
add_library(hmsbeagle-cpu MODULE
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
install(TARGETS hmsbeagle-cpu.30 hmsbeagle-cpu
        DESTINATION lib
        )