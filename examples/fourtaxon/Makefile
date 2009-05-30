CPP 		= g++
CPP_FLAGS 	= -O3 -I. -I../../include

.cpp.o:
	$(CPP) $(CPP_FLAGS) -c $*.cpp
	
fourtaxon: fourtaxon.o
	$(CPP) -o fourtaxon fourtaxon.o 
	
clean:
	rm -f *.o
	rm -f fourtaxon
	