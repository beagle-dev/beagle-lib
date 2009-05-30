#
# command-line: make STATE_COUNT=<#>
#

STATE_COUNT = 4

############


OPTIONS		:= -funroll-loops -ffast-math -fstrict-aliasing

############################################################


default :
	g++ -o tinytest -O3 -fast $(OPTIONS) $(CXX_FLAGS)	 \
		-D STATE_COUNT=$(STATE_COUNT) \
		-Iinclude/ \
		src/beagle.cpp \
		src/CPU/beagleCPUImpl.cpp \
	   	src/tests/tinyTest.cpp
