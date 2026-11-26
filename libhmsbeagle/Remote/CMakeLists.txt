# The remote plugin (the resources of beagle-worker processes on other machines appear as resources here) and
# beagle-worker itself. The two share the protocol and the transport.

set(BEAGLE_REMOTE_COMMON
        RemoteProtocol.h
        RemoteChannel.h
        RemoteChannel.cpp
        RemoteAuth.h
        RemoteAuth.cpp
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
        worker/LinkAddresses.h
        worker/LinkAddresses.cpp
        ${BEAGLE_REMOTE_COMMON}
        )

target_link_libraries(beagle-worker hmsbeagle Threads::Threads)

# finding which network interfaces are Thunderbolt links
if(APPLE)
    target_link_libraries(beagle-worker "-framework CoreFoundation" "-framework IOKit"
                          "-framework SystemConfiguration")
endif()

install(TARGETS beagle-worker
    DESTINATION bin
    COMPONENT remote
    )
