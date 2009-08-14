#
#

OUTNAME		:= hmsbeagle-jni

############

MAC_JAVA_HOME	:= /Library/Java/Home
#LINUX_JAVA_HOME	:= /usr/java/j2sdk1.4.1_01
LINUX_JAVA_HOME	:= /usr/java/jdk1.6.0_07

MAC_INCLUDES	+= -I$(MAC_JAVA_HOME)/include -I.
LINUX_INCLUDES	+= -I$(LINUX_JAVA_HOME)/include -I$(LINUX_JAVA_HOME)/include/linux -Iinclude/

DEST	:= lib

ARCH		:= i386
ARCH2		:= x86_64

MAC_LINK	:= bundle  # Can also be 'dynamiclib'
LINUX_LINK	:= shared

OPTIONS		:= -fast -funroll-loops -ffast-math -fstrict-aliasing

############################################################


mac :
	g++ -o $(DEST)/lib$(OUTNAME).$(ARCH).jnilib \
		$(OPTIONS) \
		-framework JavaVM -arch $(ARCH) \
	    -$(MAC_LINK) \
	    $(MAC_INCLUDES) \
	    -DDOUBLE_PRECISION \
 		libhmsbeagle/beagle.cpp libhmsbeagle/CPU/BeagleCPUImpl.cpp libhmsbeagle/CPU/BeagleCPU4StateImpl.cpp libhmsbeagle/JNI/beagle_BeagleJNIWrapper.cpp

	g++ -o $(DEST)/lib$(OUTNAME).$(ARCH2).jnilib \
		$(OPTIONS) \
		-framework JavaVM -arch $(ARCH2) \
	    -$(MAC_LINK) \
	    $(MAC_INCLUDES) \
	    -DDOUBLE_PRECISION \
 		libhmsbeagle/beagle.cpp libhmsbeagle/CPU/BeagleCPUImpl.cpp libhmsbeagle/CPU/BeagleCPU4StateImpl.cpp libhmsbeagle/JNI/beagle_BeagleJNIWrapper.cpp


	lipo -create $(DEST)/lib$(OUTNAME).$(ARCH).jnilib \
				$(DEST)/lib$(OUTNAME).$(ARCH2).jnilib \
	     -output $(DEST)/lib$(OUTNAME).jnilib

#	rm $(DEST)/lib$(OUTNAME).$(ARCH).jnilib $(DEST)/lib$(OUTNAME).$(ARCH2).jnilib

linux :
	gcc -c -O4 $(OPTIONS) $(LINUX_INCLUDES) -c $(INNAME) -std=c99 -DSTATE_COUNT=$(STATE_COUNT) \
	    -o $(DEST)/lib$(OUTNAME).o
	ld -$(LINUX_LINK) -o $(DEST)/lib$(OUTNAME).so $(DEST)/lib$(OUTNAME).o
	rm $(DEST)/lib$(OUTNAME).o

