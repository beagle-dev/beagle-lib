# The remote plugin (the resources of beagle-worker processes on other machines appear as resources here) and
# beagle-worker itself. The two share the protocol and the transport.

set(BEAGLE_REMOTE_COMMON
        RemoteProtocol.h
        RemoteChannel.h
        RemoteChannel.cpp
        )

add_library(hmsbeagle-remote SHARED
        BeagleRemoteImpl.h
        BeagleRemoteImpl.cpp
        BeagleRemotePlugin.h
        BeagleRemotePlugin.cpp
        ${BEAGLE_REMOTE_COMMON}
        )

install(TARGETS hmsbeagle-remote
    DESTINATION ${BEAGLE_INSTALL_DIR}
    COMPONENT remote
    )

SET_TARGET_PROPERTIES(hmsbeagle-remote
    PROPERTIES
    SOVERSION "${BEAGLE_PLUGIN_VERSION_EXTENDED}"
    SUFFIX "${BEAGLE_PLUGIN_SUFFIX}"
    )

target_compile_options(hmsbeagle-remote PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

find_package(Threads REQUIRED)

add_executable(beagle-worker
        worker/BeagleWorker.cpp
        ${BEAGLE_REMOTE_COMMON}
        )

target_link_libraries(beagle-worker hmsbeagle Threads::Threads)

install(TARGETS beagle-worker
    DESTINATION bin
    COMPONENT remote
    )
