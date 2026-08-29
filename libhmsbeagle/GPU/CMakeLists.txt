
if(BUILD_OPENCL OR BUILD_TINYGPU_HYBRID)
	# BeagleOpenCL_kernels.h is backend-agnostic text (kernelsAll.cu/kernels4.cu/
	# GPUImplDefs.h stringified as-is; see kernels/make_opencl_kernels.sh) --
	# CMake_OpenCL compiles it as OpenCL-C (FW_OPENCL); CMake_TinyGPUHybrid's AMD
	# path compiles the *same* text via comgr's HIP language instead
	# (GPUInterfaceTinyGPUHybridAMD.cpp / GPUImplDefs.h's FW_TINYGPU_HYBRID_AMD
	# branch -- STATUS.md AMD §18-19), selected entirely by which #defines get
	# prepended at compile time, not by a different generated file.
	#
	# The add_custom_command below (the actual OUTPUT-generating rule) stays
	# here, shared, guarded exactly as before: CMake only allows one such rule
	# per output file, so it can't be defined separately in both CMake_OpenCL
	# and CMake_HIP without erroring when both are enabled.
	#
	# _BeagleOpenCLKernelsFile wraps that file as a target *in this same
	# directory* -- required, not just tidy: the "Unix Makefiles" generator
	# doesn't reliably re-run an add_custom_command when a target that
	# DEPENDS on its output file directly lives in a *different* directory
	# (confirmed: building a specific consumer target after touching
	# kernelsAll.cu left the header stale). Cross-directory target-to-target
	# add_dependencies() doesn't have this problem, so CMake_OpenCL/
	# CMakeLists.txt's OpenCLKernels and CMake_HIP/CMakeLists.txt's
	# HIPKernels each add_dependencies() on this target instead of
	# depending on the file themselves.
	if(WIN32)
		set(OPENCL_KERNELS_CMD_NAME createOpenCLHeader.bat)
	else(WIN32)
		set(OPENCL_KERNELS_CMD_NAME make_opencl_kernels.sh)
	endif(WIN32)
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleOpenCL_kernels.h"
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${OPENCL_KERNELS_CMD_NAME}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS "kernels/kernels4.cu" "kernels/kernels4Derivatives.cu" "kernels/kernelsXDerivatives.cu" "kernels/kernelsAll.cu"
			COMMENT "Building OpenCL kernels"
			VERBATIM)
	add_custom_target(_BeagleOpenCLKernelsFile DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleOpenCL_kernels.h")
endif()

if(BUILD_OPENCL)
	find_package(OpenCL)
	if(OpenCL_FOUND)
		message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
		message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
		add_subdirectory("CMake_OpenCL")
	endif(OpenCL_FOUND)
endif(BUILD_OPENCL)

if(BUILD_CUDA)
	find_package(CUDA)
	if(CUDA_FOUND)
		message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
		add_subdirectory("CMake_CUDA")
	endif(CUDA_FOUND)
endif(BUILD_CUDA)

if(BUILD_TINYGPU OR BUILD_TINYGPU_HYBRID)
	# Both backends #include kernels/BeagleTinyGPU_kernels.h (see GPUInterface.h,
	# FW_TINYGPU branch) — generated once here (not per-subdirectory) so enabling
	# both doesn't register two custom-command rules for the same output file.
	# NOTE (usb branch): nvcc11 (CUDA 11.0, wrapped at ~/.local/bin/nvcc11)
	# was tried here in preference to nvcc -- its PTX generation predicates
	# kernelPartialsPartialsNoScale's guard where 12.8 branches, for
	# identical C source (STATUS.md §22) -- but caused kernelMatrixMulADB
	# (a *different*, unrelated-by-C-source kernel, with IDENTICAL SASS
	# branch structure either way) to hang on real hardware (STATUS.md
	# §23/24). Reverted to plain nvcc (12.8), the known-safe compiler.
	# nvcc11/nvcc11shim left in place at ~/.local/bin, unused.
	find_program(TINYGPU_NVCC NAMES nvcc HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	if(NOT TINYGPU_NVCC)
		message(FATAL_ERROR "nvcc not found (required to build kernels/BeagleTinyGPU_kernels.h "
			"for the TinyGPU/TinyGPUHybrid backend). Add nvcc to PATH, or pass "
			"-DTINYGPU_NVCC=/path/to/nvcc.")
	endif()
	message(STATUS "TinyGPU kernels: using nvcc at ${TINYGPU_NVCC}")
	set(TINYGPU_NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets")
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h"
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_kernels.sh" "${TINYGPU_NVCC}" "${TINYGPU_NVCCFLAGS}" "-I${CMAKE_SOURCE_DIR}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS "kernels/kernels4.cu" "kernels/kernelsX.cu" "kernels/kernelsAll.cu" "kernels/make_tinygpu_kernels.sh"
			COMMENT "Building TinyGPU-specific kernel PTX (-DCUDA -DFW_TINYGPU)"
			VERBATIM)
	add_custom_target(TinyGPUKernels DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h")
endif()

if(BUILD_TINYGPU)
	message(STATUS "TinyGPU backend enabled (USB4/Thunderbolt eGPU via TinyGPU.app)")
	add_subdirectory("CMake_TinyGPU")
endif(BUILD_TINYGPU)

if(BUILD_TINYGPU_HYBRID)
	message(STATUS "TinyGPUHybrid backend enabled (tinygrad init + C++ dispatch)")
	# CMake_HIP only defines the HIPKernels target (see the BeagleOpenCL_kernels.h
	# comment above) -- added before CMake_TinyGPUHybrid so that target exists
	# by the time CMake_TinyGPUHybrid/CMakeLists.txt's add_dependencies needs it.
	add_subdirectory("CMake_HIP")
	add_subdirectory("CMake_TinyGPUHybrid")
endif(BUILD_TINYGPU_HYBRID)
