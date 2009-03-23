#
# command-line: make STATE_COUNT=<#>
#

STATE_COUNT = 4

############


OPTIONS		:= -funroll-loops -ffast-math -fstrict-aliasing

############################################################


default :
	cc -o tinytest -O3 -fast $(OPTIONS) -std=c99 \
		-D STATE_COUNT=$(STATE_COUNT) \
		-Iinclude/ \
		src/CPU/beagleCPU2.c \
	   	src/tests/tinyTest.c
