noinst_LTLIBRARIES=libcpu.la 

libcpu_la_SOURCES = BeagleCPUImpl.cpp BeagleCPUImpl.h
libcpu_la_LDFLAGS = 
# TODO: remove CUDA conditional here, should not be necessary
if BUILDCUDA
libcpu_la_CFLAGS = $(CUDA_CFLAGS)
libcpu_la_CXXFLAGS = $(CUDA_CFLAGS)
libcpu_la_LIBADD = $(CUDA_LIBS) 
endif

INCLUDES = -I$(top_builddir) -I$(top_srcdir)
