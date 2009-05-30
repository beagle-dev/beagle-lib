CPP 		= g++
CPP_FLAGS 	= -O3 -I. -I../include

.cpp.o:
	$(CPP) $(CPP_FLAGS) -c $*.cpp
	
example: example.o
	$(CPP) -o example example.o 
	
clean:
	rm -f *.o
	rm example
	