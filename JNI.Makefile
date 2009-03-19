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

ARCH		:= i386

MAC_LINK	:= bundle  # Can also be 'dynamiclib'
LINUX_LINK	:= shared

OPTIONS		:= -funroll-loops -ffast-math -fstrict-aliasing

############################################################


mac :
	cc -c -arch $(ARCH) -O3 -fast $(OPTIONS) $(MAC_INCLUDES) -std=c99 \
	   -DSTATE_COUNT=$(STATE_COUNT) \
	   -DDOUBLE_PRECISION \
	   -o $(OUTNAME).$(ARCH).o src/CPU/beagleCPU.c java/JNI/beagle_BeagleJNIWrapper.c
	cc -o lib$(OUTNAME).$(ARCH).jnilib -framework JavaVM -arch $(ARCH) \
	   -$(MAC_LINK) $(OUTNAME).$(ARCH).o
	lipo -create lib$(OUTNAME).$(ARCH).jnilib \
	     -output lib$(OUTNAME).jnilib

linux :
	gcc -c -O4 $(OPTIONS) $(LINUX_INCLUDES) -c $(INNAME) -std=c99 -DSTATE_COUNT=$(STATE_COUNT)  -o lib$(OUTNAME).o
	ld -$(LINUX_LINK) -o lib$(OUTNAME).so lib$(OUTNAME).o

