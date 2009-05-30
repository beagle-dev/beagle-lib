#
# command-line: make STATE_COUNT=<#>
#

STATE_COUNT = 4

############


#OPTIONS		:= -O0 -fast  -funroll-loops -ffast-math -fstrict-aliasing

############################################################



default :
	g++ -o tinytest $(OPTIONS) $(CXX_FLAGS)	 \
		-D STATE_COUNT=$(STATE_COUNT) \
		-Iinclude/ \
		src/beagle.cpp \
		src/CPU/beagleCPUImpl.cpp \
	   	src/tests/tinyTest.cpp
