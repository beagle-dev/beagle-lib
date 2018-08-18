# Alright, this cmakelists.txt file is used to create beagle library

# build minimum functional beagle-lib
add_library(hmsbeagle SHARED
        beagle.cpp
        beagle.h
        BeagleImpl.h
        config.h
        platform.h

        benchmark/BeagleBenchmark.h
        benchmark/BeagleBenchmark.cpp
        benchmark/linalg.h
        benchmark/linalg.cpp

#        JNI/beagle_BeagleJNIWrapper.cpp
#        JNI/beagle_BeagleJNIWrapper.h

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
add_subdirectory(JNI)

install(TARGETS hmsbeagle
        DESTINATION lib
        )

# Now set installation directory
#install(TARGETS beagle_lib
#        ARCHIVE DESTINATION .libs
#        LIBRARY DESTINATION .libs
#        COMPONENT library)
