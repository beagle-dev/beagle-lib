# clang++ -dynamiclib -undefined dynamic_lookup -o libhmsbeagle-jni.jnilib beagle_BeagleJNIWrapper.o
# cc -bundle -l hmsbeagle -o libhmsbeagle-jni.jnilib beagle_BeagleJNIWrapper.o

add_library(hmsbeagle-jni SHARED
        beagle_BeagleJNIWrapper.cpp
        beagle_BeagleJNIWrapper.h
        )

target_link_libraries(hmsbeagle-jni hmsbeagle)

if(APPLE)
    set_target_properties(hmsbeagle-jni PROPERTIES PREFIX "lib" SUFFIX ".jnilib")
elseif(WIN32)
    set_target_properties(hmsbeagle-jni PROPERTIES PREFIX "" SUFFIX "64.dll")
elseif(UNIX)
    set_target_properties(hmsbeagle-jni PROPERTIES PREFIX "lib" SUFFIX ".so")
endif(APPLE)

install(TARGETS hmsbeagle-jni
        DESTINATION ${BEAGLE_INSTALL_DIR}
        )

if(BUILD_BASTA)
	add_library(hmsbeagle-jni-basta SHARED
		beagle_basta_BastaJNIWrapper.cpp
		beagle_basta_BastaJNIWrapper.h
		)
		
	target_link_libraries(hmsbeagle-jni-basta hmsbeagle)
	
	if(APPLE)
    	set_target_properties(hmsbeagle-jni-basta PROPERTIES PREFIX "lib" SUFFIX ".jnilib")
	elseif(WIN32)
    	set_target_properties(hmsbeagle-jni-basta PROPERTIES PREFIX "" SUFFIX "64.dll")
	elseif(UNIX)
    	set_target_properties(hmsbeagle-jni-basta PROPERTIES PREFIX "lib" SUFFIX ".so")
	endif(APPLE)	
	
	install(TARGETS hmsbeagle-jni-basta
    	    DESTINATION ${BEAGLE_INSTALL_DIR}
        	)
	
endif(BUILD_BASTA)
