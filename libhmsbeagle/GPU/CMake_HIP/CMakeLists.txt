
# CMake_HIP — not a build target of its own. Defines HIPKernels, the target
# CMake_TinyGPUHybrid's AMD path depends on for the kernel source it compiles
# via comgr's HIP language (GPUInterfaceTinyGPUHybridAMD.cpp / GPUImplDefs.h's
# FW_TINYGPU_HYBRID_AMD branch -- STATUS.md AMD §18-19).
#
# BeagleOpenCL_kernels.h's content is backend-agnostic text (kernelsAll.cu/
# kernels4.cu/GPUImplDefs.h stringified as-is); which language it's compiled
# as is decided entirely by #defines prepended at compile time
# (amd_compile_helper.compile_hip()), not by a different generated file. Its
# own generation rule (add_custom_command) lives in the parent
# libhmsbeagle/GPU/CMakeLists.txt, shared with CMake_OpenCL (see that file's
# comment) since a given output file can only have one generating rule. This
# target wraps _BeagleOpenCLKernelsFile (also defined there, in the same
# directory as the generating rule -- required for the Makefile generator to
# actually re-run it, not just DEPENDS on the file directly here) under a
# name specific to this backend.
add_custom_target(HIPKernels)
add_dependencies(HIPKernels _BeagleOpenCLKernelsFile)
