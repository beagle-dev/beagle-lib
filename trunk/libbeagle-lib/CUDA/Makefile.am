# rules for building cuda files
.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) -DHAVE_CONFIG_H -I$(top_srcdir)
.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -I$(top_srcdir) -c $(NVCCFLAGS) -DHAVE_CONFIG_H $<

CCLD = g++

if BUILDCUDA
noinst_LTLIBRARIES= libcuda.la 
libcuda_la_SOURCES = \
BeagleCUDAImpl.cpp  CUDASharedFunctions.c   CUDASharedFunctions.h  PeelingFunctions.cu \
TransitionFunctions.cu  BeagleCUDAImpl.h

libcuda_la_CFLAGS = $(CUDA_CFLAGS)
libcuda_la_CXXFLAGS = $(CUDA_CFLAGS)
libcuda_la_LIBADD = $(CUDA_LIBS) 
libcuda_la_LDFLAGS = 

libcuda_la_LINK = $(LIBTOOL) --tag=CC $(AM_LIBTOOLFLAGS) \
        $(LIBTOOLFLAGS) --mode=link $(CCLD) $(libcuda_la_CFLAGS) \
        $(CFLAGS) $(libcuda_la_LDFLAGS) $(LDFLAGS) -o $@
endif

INCLUDES = -I$(top_builddir) -I$(top_srcdir)

