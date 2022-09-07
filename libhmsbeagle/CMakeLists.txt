if(WIN32)
	add_definitions(-D_EXPORTING)
	add_definitions(-DWIN32)
	add_definitions(-D_CONSOLE)
#	add_definitions(-DPACKAGE_VERSION="${BEAGLE_VERSION}")
#	add_definitions(-DPLUGIN_VERSION="${BEAGLE_PLUGIN_VERSION}")

	set(BEAGLE_PLUGIN_SUFFIX "64-${BEAGLE_PLUGIN_VERSION}.dll")
    set(BEAGLE_INSTALL_DIR ".")
    set(BEAGLE_HEADER_DIR ".")

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
    set(BEAGLE_HEADER_DIR include/libhmsbeagle-1/libhmsbeagle)

    set(BEAGLE_PLUGIN_SOURCE
        plugin/UnixSharedLibrary.cpp
	    plugin/UnixSharedLibrary.h
		plugin/LibtoolSharedLibrary.h
        )

    set(BEAGLE_INTEGRATED_JNI_SOURCE
        )

	if(BEAGLE_EXPERIMENTAL_LTDL)
	    find_library(BEAGLE_LTDL_LIB
    		NAMES ltdl
	    	PATH_SUFFIXES lib lib64)
	endif(BEAGLE_EXPERIMENTAL_LTDL)

    if(BEAGLE_LTDL_LIB)
    	add_definitions(-DHAVE_LIBLTDL)
    	message(STATUS "Using libtools for plugins: ${BEAGLE_LTDL_LIB}")
    else(BEAGLE_LTDL_LIB)
    	message(STATUS "Not using libtools for plugins")
    endif(BEAGLE_LTDL_LIB)

    if(NOT APPLE)
    	set(BEAGLE_PLUGIN_VERSION_EXTENDED "${BEAGLE_PLUGIN_VERSION}.0.0")
    else(NOT APPLE)
    	set(BEAGLE_PLUGIN_VERSION_EXTENDED "${BEAGLE_PLUGIN_VERSION}")
    endif(NOT APPLE)

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
        benchmark/BenchmarkHelper.h

        ${BEAGLE_INTEGRATED_JNI_SOURCE}

        plugin/BeaglePlugin.h
        plugin/Plugin.cpp
        plugin/Plugin.h
        plugin/SharedLibrary.h
        ${BEAGLE_PLUGIN_SOURCE}
        )

if(NOT WIN32)
	if(NOT APPLE)
		target_link_libraries(hmsbeagle ${CMAKE_DL_LIBS})
	endif(NOT APPLE)
endif(NOT WIN32)

if (BUILD_ACTION)
	target_link_libraries(hmsbeagle Eigen3::Eigen)
endif(BUILD_ACTION)

set_target_properties(hmsbeagle PROPERTIES PUBLIC_HEADER "beagle.h;platform.h")

add_subdirectory(CPU)
add_subdirectory(GPU)
if(BUILD_JNI)
    add_subdirectory(JNI)
endif(BUILD_JNI)

install(TARGETS hmsbeagle
	LIBRARY DESTINATION ${BEAGLE_INSTALL_DIR}
	PUBLIC_HEADER DESTINATION ${BEAGLE_HEADER_DIR}
	COMPONENT main
    )

if(WIN32)
    SET_TARGET_PROPERTIES(hmsbeagle
        PROPERTIES
	    SUFFIX "64.dll"
        )
endif(WIN32)
