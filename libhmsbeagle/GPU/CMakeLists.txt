
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
			DEPENDS "kernels/kernels4.cu" "kernels/kernelsX.cu" "kernels/kernels4Derivatives.cu" "kernels/kernelsXDerivatives.cu" "kernels/kernelsAll.cu" "GPUImplDefs.h"
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
	# Plain nvcc (12.8). An nvcc11 build once hung kernelMatrixMulADB
	# (STATUS.md §24), before the launch-dims fix (STATUS.md §203); not
	# re-tested since.
	find_program(TINYGPU_NVCC NAMES nvcc HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	if(NOT TINYGPU_NVCC)
		message(FATAL_ERROR "nvcc not found (required to build kernels/BeagleTinyGPU_kernels.h "
			"for the TinyGPU/TinyGPUHybrid backend). Add nvcc to PATH, or pass "
			"-DTINYGPU_NVCC=/path/to/nvcc.")
	endif()
	message(STATUS "TinyGPU kernels: using nvcc at ${TINYGPU_NVCC}")
	set(TINYGPU_NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets")
	# The script also keeps the SP modules for the TinyGPUHybrid cubins (TODO.md
	# plan step C1). Declared as outputs, so a missing one reruns the script.
	set(TINYGPU_SP_STATES 4 16 32 48 64 80 128 192 256)   # as in make_tinygpu_kernels.sh and make_tinygpu_cubins.sh
	foreach(s ${TINYGPU_SP_STATES})
		list(APPEND TINYGPU_SP_PTX "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tinygpu_cubins/SP_${s}.ptx")
	endforeach()
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h" ${TINYGPU_SP_PTX}
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_kernels.sh" "${TINYGPU_NVCC}" "${TINYGPU_NVCCFLAGS}" "-I${CMAKE_SOURCE_DIR}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS "kernels/kernels4.cu" "kernels/kernelsX.cu" "kernels/kernelsAll.cu" "kernels/kernels4Derivatives.cu" "kernels/kernelsXDerivatives.cu" "GPUImplDefs.h" "kernels/make_tinygpu_kernels.sh"
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
	# The C++ runtime's ahead-of-time cubins (TODO.md plan step C1, decision
	# 15): the SP modules kept above, compiled with ptxas for sm_86, sm_89 and
	# sm_120 and embedded in the plugin by kernels/BeagleTinyGPU_cubins.S.
	# There is no run-time fallback, so ptxas is required like nvcc. The
	# add_dependencies is needed, not just tidy: the TinyGPUKernels rule joins
	# TinyGPUCubins through the PTX files, and two parallel runs of
	# make_tinygpu_kernels.sh (make -j) corrupt the header.
	find_program(TINYGPU_PTXAS NAMES ptxas HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	if(NOT TINYGPU_PTXAS)
		message(FATAL_ERROR "ptxas not found (required to build kernels/BeagleTinyGPU_cubins.S "
			"for the TinyGPUHybrid backend). Add ptxas to PATH, or pass "
			"-DTINYGPU_PTXAS=/path/to/ptxas.")
	endif()
	message(STATUS "TinyGPU cubins: using ptxas at ${TINYGPU_PTXAS}")
	# The cubins are outputs too (a missing one reruns the script) and, in
	# CMake_TinyGPUHybrid, dependencies of the .S's object (.incbin is not
	# scanned, so a changed cubin would not be re-embedded otherwise).
	foreach(s ${TINYGPU_SP_STATES})
		foreach(a sm_86 sm_89 sm_120)   # as in make_tinygpu_cubins.sh
			list(APPEND TINYGPU_CUBINS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tinygpu_cubins/SP_${s}_${a}.cubin")
		endforeach()
	endforeach()
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_cubins.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUNVCubins.h" ${TINYGPU_CUBINS}
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_cubins.sh" "${TINYGPU_PTXAS}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS ${TINYGPU_SP_PTX} "kernels/make_tinygpu_cubins.sh"
			COMMENT "Building TinyGPU ahead-of-time cubins (ptxas: SP x sm_86, sm_89, sm_120)"
			VERBATIM)
	add_custom_target(TinyGPUCubins DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_cubins.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUNVCubins.h")
	add_dependencies(TinyGPUCubins TinyGPUKernels)
	# CMake_HIP only defines the HIPKernels target (see the BeagleOpenCL_kernels.h
	# comment above) -- added before CMake_TinyGPUHybrid so that target exists
	# by the time CMake_TinyGPUHybrid/CMakeLists.txt's add_dependencies needs it.
	add_subdirectory("CMake_HIP")
	add_subdirectory("CMake_TinyGPUHybrid")
endif(BUILD_TINYGPU_HYBRID)
