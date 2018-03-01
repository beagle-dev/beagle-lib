# Alright, this cmakelists.txt file is used to create beagle library

# build minimum functional beagle-lib
add_library(beagle_lib SHARED
        beagle.cpp
        beagle.h
        BeagleImpl.h
        config.h
        platform.h

#        CPU/BeagleCPU4StateImpl.h
#        CPU/BeagleCPU4StateImpl.hpp

#        CPU/BeagleCPUImpl.h
#        CPU/BeagleCPUImpl.hpp
#        CPU/BeagleCPUPlugin.cpp
#        CPU/BeagleCPUPlugin.h
#        CPU/EigenDecomposition.h
#        CPU/EigenDecompositionCube.h
#        CPU/EigenDecompositionCube.hpp
#        CPU/EigenDecompositionSquare.h
#        CPU/EigenDecompositionSquare.hpp
#        CPU/Precision.h
#        CPU/SSEDefinitions.h

        JNI/beagle_BeagleJNIWrapper.cpp
        JNI/beagle_BeagleJNIWrapper.h
        plugin/BeaglePlugin.h
        plugin/LibtoolSharedLibrary.h
        plugin/Plugin.cpp
        plugin/Plugin.h
        plugin/SharedLibrary.h
        plugin/UnixSharedLibrary.cpp
        plugin/UnixSharedLibrary.h
        )

add_subdirectory(CPU)
#add_subdirectory(GPU)
#add_subdirectory(JNI)


# Now set installation directory
#install(TARGETS beagle_lib
#        ARCHIVE DESTINATION .libs
#        LIBRARY DESTINATION .libs
#        COMPONENT library)
