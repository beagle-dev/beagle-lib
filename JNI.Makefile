#
# command-line: make STATE_COUNT=<#> mac/linux
#

STATE_COUNT = 4

OUTNAME		:= BEAGLE-$(STATE_COUNT)-D

############

MAC_JAVA_HOME	:= /Library/Java/Home
#LINUX_JAVA_HOME	:= /usr/java/j2sdk1.4.1_01
LINUX_JAVA_HOME	:= /usr/java/jdk1.6.0_07

MAC_INCLUDES	+= -I$(MAC_JAVA_HOME)/include -Iinclude/
LINUX_INCLUDES	+= -I$(LINUX_JAVA_HOME)/include -I$(LINUX_JAVA_HOME)/include/linux -Iinclude/

DEST	:= lib

ARCH		:= i386

MAC_LINK	:= bundle  # Can also be 'dynamiclib'
LINUX_LINK	:= shared

OPTIONS		:= -fast
# -fast -ffast-math -funroll-loops -fstrict-aliasing

############################################################


mac :
	cc -c -arch $(ARCH) -O3 $(OPTIONS) $(MAC_INCLUDES) -std=c99 \
	   -DSTATE_COUNT=$(STATE_COUNT) \
	   -DDOUBLE_PRECISION \
	   -o $(DEST)/beagleCPU2.$(ARCH).o src/CPU/beagleCPU2.c 
	cc -c -arch $(ARCH) -O3 $(OPTIONS) $(MAC_INCLUDES) -std=c99 \
	   -DSTATE_COUNT=$(STATE_COUNT) \
	   -DDOUBLE_PRECISION \
	   -o $(DEST)/beagle_BeagleJNIWrapper.$(ARCH).o java/JNI/beagle_BeagleJNIWrapper.c
	cc -o $(DEST)/lib$(OUTNAME).$(ARCH).jnilib -framework JavaVM -arch $(ARCH) \
	   -$(MAC_LINK) $(DEST)/beagleCPU2.$(ARCH).o $(DEST)/beagle_BeagleJNIWrapper.$(ARCH).o
	lipo -create $(DEST)/lib$(OUTNAME).$(ARCH).jnilib \
	     -output $(DEST)/lib$(OUTNAME).jnilib
	rm $(DEST)/*.o $(DEST)/lib$(OUTNAME).$(ARCH).jnilib

linux :
	gcc -c -O4 $(OPTIONS) $(LINUX_INCLUDES) -c $(INNAME) -std=c99 -DSTATE_COUNT=$(STATE_COUNT) \
	    -o $(DEST)/lib$(OUTNAME).o
	ld -$(LINUX_LINK) -o $(DEST)/lib$(OUTNAME).so $(DEST)/lib$(OUTNAME).o
	rm $(DEST)/lib$(OUTNAME).o

