# rules for building cuda files
.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) -I$(top_srcdir)
.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -I$(top_srcdir) -c $(NVCCFLAGS) $<

CCLD = g++

if BUILDCUDA
noinst_LTLIBRARIES=libcuda.la 
libcuda_la_SOURCES = \
CUDASharedFunctions_kernel.cu  TransitionProbabilities_kernel.cu Peeling_kernel.cu \
BeagleCUDATipStateImpl.cpp  CUDASharedFunctions.c  Queue.cpp \
BeagleCUDATipStateImpl.h  CUDASharedFunctions.h  Queue.h 
libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)
libcuda_la_LIBADD = $(CUDA_LIBS) 
libcuda_la_LDFLAGS = 

libcuda_la_LINK = $(LIBTOOL) --tag=CC $(AM_LIBTOOLFLAGS) \
        $(LIBTOOLFLAGS) --mode=link $(CCLD) $(libdcuda_la_CFLAGS) \
        $(CFLAGS) $(libdcuda_la_LDFLAGS) $(LDFLAGS) -o $@
endif

INCLUDES = -I$(top_builddir)

