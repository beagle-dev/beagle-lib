#
# command-line: make STATE_COUNT=<#>
# 

OSNAME := $(shell uname)

ifeq ($(OSNAME),Linux)
	EXT				:= so
	CUDA_SDK_PATH	:= /opt/NVIDIA_CUDA_SDK
	JAVA_HOME 		:= /usr/java/default
	INCLUDES		+= -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -Iinclude
endif

ifeq ($(OSNAME),Darwin)
	EXT 			:=	jnilib
	CUDA_SDK_PATH 	:= /Developer/CUDA
	JAVA_HOME		:= /System/Library/Frameworks/JavaVM.framework
	INCLUDES		+= -I$(JAVA_HOME)/Headers -Iinclude
endif

CUDA_INSTALL_PATH	:= /usr/local/cuda

BASEDIR=.
TARGET_DIR=$(BASEDIR)/lib
HEADER=$(BASEDIR)

STATE_COUNT = 60
SCALE = 1E+0f
		       
EXECUTABLE	:= $(TARGET_DIR)/libBEAGLE-$(STATE_COUNT).$(EXT)

CUFILES	:= ./src/CUDA/CUDASharedFunctions.c	\
		   ./src/CUDA/CUDASharedFunctions_kernel.cu \
		   ./src/CUDA/beagleCUDA.c \
		   ./src/CUDA/TransitionProbabilities_kernel.cu \
		   ./src/CUDA/Peeling_kernel.cu \
		   ./java/JNI/beagle_BeagleJNIWrapper.c
		   
CUFILES_CPP	:=	src/CUDA/CUDASharedFunctions.c	\
				src/CUDA/Queue.cpp \
				src/CUDA/CUDASharedFunctions_kernel.cu \
				src/CUDA/BeagleCUDAImpl.cpp \
				src/CUDA/TransitionProbabilities_kernel.cu \
				src/CUDA/Peeling_kernel.cu \
				src/beagle.cpp \
				java/JNI/beagle_BeagleJNIWrapper.cpp \
				lib/BeagleCPUImpl.o	   
		   		   
OPTIONS		:= -fast -funroll-loops -ffast-math -fstrict-aliasing
		         		       		      
############################################################

# Machine-dependent includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/common/inc -I$(COMMONDIR)/inc \
	  	     -I$(HEADER) -I$(TARGET_DIR)classes 
# Libs
LIB       := -L$(CUDA_INSTALL_PATH)/common/lib
DOLINK    := -lcuda -lcudart

all : cpp-device

directories :
	if test ! -e $(TARGET_DIR); then mkdir $(TARGET_DIR); fi

clean:
	rm -f $(CCOFILES) $(EXECUTABLE)
	
cpp-device: directories cpu
	@echo "using device mode!"
	nvcc -O4 -shared -DSTATE_COUNT=$(STATE_COUNT) \
		 --compiler-options -fPIC \
		 --compiler-options -funroll-loops \
		 -DCUDA \
		 -o $(EXECUTABLE)  $(CUFILES_CPP) $(INCLUDES) $(LIB) $(DOLINK)
		
cpp-debug: directories cpu
	@echo "using debug mode!"
	nvcc -O4 -shared -DSTATE_COUNT=$(STATE_COUNT) \
		 --compiler-options -fPIC \
		 --compiler-options -funroll-loops \
		 -DCUDA \
		 -DDEBUG \
		 -o $(EXECUTABLE)  $(CUFILES_CPP) $(INCLUDES) $(LIB) $(DOLINK)	
		 
cpp-device-double: directories cpu
	@echo "using device mode!"
	nvcc -O4 -shared -DSTATE_COUNT=$(STATE_COUNT) \
		 --compiler-options -fPIC \
		 --compiler-options -funroll-loops \
		 -DCUDA \
		 -arch sm_13 -DDOUBLE_PRECISION \
		 -o $(EXECUTABLE)  $(CUFILES_CPP) $(INCLUDES) $(LIB) $(DOLINK)
		 	
#emulation-double: directories
#	@echo "using emulation mode!"
#	nvcc -g -DDEBUG -DKERNEL_PRINT_ENABLED -O3 -shared -DSTATE_COUNT=$(STATE_COUNT) -DSCALE=$(SCALE)\
#	     -arch sm_13 -DDOUBLE_PRECISION \
#		 -o $(EXECUTABLE) $(CUFILES) $(INCLUDES) $(lib) $(DOLINK) -deviceemu		 
	
cpu : BeagleCPUImpl.o


BeagleCPUImpl.o :
	g++ -c -o lib/BeagleCPUImpl.o \
		$(OPTIONS) \
	    -DSTATE_COUNT=$(STATE_COUNT) \
	    -DDOUBLE_PRECISION \
		$(INCLUDES) \
		$(LIB) \
 		src/CPU/BeagleCPUImpl.cpp 	



