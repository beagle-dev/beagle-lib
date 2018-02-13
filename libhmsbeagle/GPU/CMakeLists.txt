add_library(hmsbeagle-gpu.30 MODULE
#        BeagleCPU4StateImpl.h
#        BeagleCPU4StateImpl.hpp
        BeagleGPUImpl.h
        BeagleGPUImpl.hpp
        CUDAPlugin.cpp
        CUDAPlugin.h
        GPUInterface.h
        GPUImplDefs.h
        GPUImplHelper.h
        KernelLauncher.h
        KernelResource.h
        OpenCLAlteraPlugin.h
        OpenCLPlugin.h
        Precision.h
 )