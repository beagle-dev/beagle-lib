CPP 		= g++
CPP_FLAGS 	= -g -O0 -I. -I../../include
LD_FLAGS	= -L../../lib/

.cpp.o:
	$(CPP) $(CPP_FLAGS) -c $*.cpp
	
tinytest: tinytest.o ../../src/beagle.o
	$(CPP) -o tinytest beagle.o tinytest.o ../../lib/BeagleCPUImpl.o
	
clean:
	rm -f *.o
	rm -f tinytest
	