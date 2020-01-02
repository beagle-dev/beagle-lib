# Alright, this cmakelists.txt file is used to create beagle library

if(WIN32)
	add_definitions(-D_EXPORTING)
	add_definitions(-DWIN32)
	add_definitions(-D_CONSOLE)
	add_definitions(-DPACKAGE_VERSION="${BEAGLE_VERSION}")
	add_definitions(-DPLUGIN_VERSION="${BEAGLE_VERSION_PLUGIN}")

	set(BEAGLE_PLUGIN_SUFFIX "64-${BEAGLE_VERSION_PLUGIN}.dll")
    set(BEAGLE_INSTALL_DIR ".")

    set(BEAGLE_PLUGIN_SOURCE
    	plugin/WinSharedLibrary.cpp
	    plugin/WinSharedLibrary.h
        )

    set(BEAGLE_INTEGRATED_JNI_SOURCE
        JNI/beagle_BeagleJNIWrapper.cpp
        JNI/beagle_BeagleJNIWrapper.h
        )
else(WIN32)
	set(BEAGLE_PLUGIN_SUFFIX ".so")
    set(BEAGLE_INSTALL_DIR lib)

    set(BEAGLE_PLUGIN_SOURCE
        plugin/UnixSharedLibrary.cpp
	    plugin/UnixSharedLibrary.h
        )

    set(BEAGLE_INTEGRATED_JNI_SOURCE
        )
endif(WIN32)

# build minimum functional beagle-lib
add_library(hmsbeagle SHARED
        beagle.cpp
        beagle.h
        BeagleImpl.h
        platform.h

        benchmark/BeagleBenchmark.h
        benchmark/BeagleBenchmark.cpp
        benchmark/linalg.h
        benchmark/linalg.cpp

        ${BEAGLE_INTEGRATED_JNI_SOURCE}

        plugin/BeaglePlugin.h
        plugin/LibtoolSharedLibrary.h
        plugin/Plugin.cpp
        plugin/Plugin.h
        plugin/SharedLibrary.h
        ${BEAGLE_PLUGIN_SOURCE}
        )

add_subdirectory(CPU)
add_subdirectory(GPU)
if(NOT WIN32)
    add_subdirectory(JNI)
endif(NOT WIN32)

install(TARGETS hmsbeagle
	DESTINATION ${BEAGLE_INSTALL_DIR}
    )

if(WIN32)
    SET_TARGET_PROPERTIES(hmsbeagle
        PROPERTIES
	    SUFFIX "64.dll"
        )
endif(WIN32)
