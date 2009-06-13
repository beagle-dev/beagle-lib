CPP 		= g++
CPP_FLAGS 	= -g -O0 -I. -I../../include
LD_FLAGS	= -L../../lib/

.cpp.o:
	$(CPP) $(CPP_FLAGS) -c $*.cpp
	
fourtaxon: fourtaxon.o ../../src/beagle.o
	$(CPP) -o fourtaxon beagle.o fourtaxon.o ../../lib/BeagleCPUImpl.o
	
clean:
	rm -f *.o
	rm -f fourtaxon
	