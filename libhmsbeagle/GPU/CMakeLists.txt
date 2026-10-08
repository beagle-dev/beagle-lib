
if(BUILD_OPENCL OR BUILD_TINYGPU)
	# BeagleOpenCL_kernels.h is backend-agnostic text (kernelsAll.cu/kernels4.cu/
	# GPUImplDefs.h stringified as-is; see kernels/make_opencl_kernels.sh) --
	# CMake_OpenCL compiles it as OpenCL-C (FW_OPENCL); CMake_TinyGPU's AMD
	# path compiles the *same* text via comgr's HIP language instead
	# (GPUInterfaceTinyGPUAMD.cpp / GPUImplDefs.h's FW_TINYGPU_AMD
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

if(BUILD_TINYGPU)
	# kernels/BeagleTinyGPU_kernels.h, generated here, is the PTX the cubins
	# below are compiled from. The plugin includes only its stamp
	# (kernels/BeagleTinyGPU_kernels_stamp.h, see GPUInterface.h's FW_TINYGPU
	# branch), not its PTX (TODO.md plan step C13, decision 15).
	# Plain nvcc (12.8). An nvcc11 build once hung kernelMatrixMulADB
	# (STATUS.md §24), before the launch-dims fix (STATUS.md §203); not
	# re-tested since.
	find_program(TINYGPU_NVCC NAMES nvcc HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	if(NOT TINYGPU_NVCC)
		message(FATAL_ERROR "nvcc not found (required to build kernels/BeagleTinyGPU_kernels.h "
			"for the TinyGPU backend). Add nvcc to PATH, or pass "
			"-DTINYGPU_NVCC=/path/to/nvcc.")
	endif()
	message(STATUS "TinyGPU kernels: using nvcc at ${TINYGPU_NVCC}")
	set(TINYGPU_NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets")
	# The script also keeps the SP and DP modules for the TinyGPU cubins
	# (TODO.md plan steps C1, C16). Declared as outputs, so a missing one reruns
	# the script.
	set(TINYGPU_SP_STATES 4 16 32 48 64 80 128 192 256)   # as in make_tinygpu_kernels.sh and make_tinygpu_cubins.sh
	foreach(p SP DP)
		foreach(s ${TINYGPU_SP_STATES})
			list(APPEND TINYGPU_PTX "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tinygpu_cubins/${p}_${s}.ptx")
		endforeach()
	endforeach()
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h" ${TINYGPU_PTX}
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_kernels.sh" "${TINYGPU_NVCC}" "${TINYGPU_NVCCFLAGS}" "-I${CMAKE_SOURCE_DIR}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS "kernels/kernels4.cu" "kernels/kernelsX.cu" "kernels/kernelsAll.cu" "kernels/kernels4Derivatives.cu" "kernels/kernelsXDerivatives.cu" "GPUImplDefs.h" "kernels/make_tinygpu_kernels.sh"
			COMMENT "Building TinyGPU-specific kernel PTX (-DCUDA -DFW_TINYGPU)"
			VERBATIM)
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels_stamp.h"
			COMMAND sh -c "grep -q '^#define TINYGPU_KERNELS_STAMP ' BeagleTinyGPU_kernels.h && { echo '// auto-generated: the stamp of BeagleTinyGPU_kernels.h, the PTX the embedded cubins are compiled from'; grep '^#define TINYGPU_KERNELS_STAMP ' BeagleTinyGPU_kernels.h; } > BeagleTinyGPU_kernels_stamp.h"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h"
			COMMENT "Extracting the TinyGPU kernel header's stamp"
			VERBATIM)
	add_custom_target(TinyGPUKernels DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels.h"
			"${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_kernels_stamp.h")
endif()

if(BUILD_TINYGPU)
	message(STATUS "TinyGPU backend enabled (NV and AMD: the C++ boot, no Python)")
	# The C++ runtime's ahead-of-time cubins (TODO.md plan steps C1 and C16,
	# decision 15): the SP and DP modules kept above, compiled with ptxas for sm_86, sm_89 and
	# sm_120 and embedded in the plugin by kernels/BeagleTinyGPU_cubins.S.
	# There is no run-time fallback, so ptxas is required like nvcc. The
	# add_dependencies is needed, not just tidy: the TinyGPUKernels rule joins
	# TinyGPUCubins through the PTX files, and two parallel runs of
	# make_tinygpu_kernels.sh (make -j) corrupt the header.
	find_program(TINYGPU_PTXAS NAMES ptxas HINTS "$ENV{HOME}/.local/bin" "/usr/local/cuda/bin")
	if(NOT TINYGPU_PTXAS)
		message(FATAL_ERROR "ptxas not found (required to build kernels/BeagleTinyGPU_cubins.S "
			"for the TinyGPU backend). Add ptxas to PATH, or pass "
			"-DTINYGPU_PTXAS=/path/to/ptxas.")
	endif()
	message(STATUS "TinyGPU cubins: using ptxas at ${TINYGPU_PTXAS}")
	# The cubins are outputs too (a missing one reruns the script) and, in
	# CMake_TinyGPU, dependencies of the .S's object (.incbin is not
	# scanned, so a changed cubin would not be re-embedded otherwise).
	foreach(p SP DP)
		foreach(s ${TINYGPU_SP_STATES})
			foreach(a sm_86 sm_89 sm_120)   # as in make_tinygpu_cubins.sh
				list(APPEND TINYGPU_CUBINS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tinygpu_cubins/${p}_${s}_${a}.cubin")
			endforeach()
		endforeach()
	endforeach()
	add_custom_command(
			OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_cubins.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUNVCubins.h" ${TINYGPU_CUBINS}
			COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_cubins.sh" "${TINYGPU_PTXAS}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
			DEPENDS ${TINYGPU_PTX} "kernels/make_tinygpu_cubins.sh"
			COMMENT "Building TinyGPU ahead-of-time cubins (ptxas: SP and DP x sm_86, sm_89, sm_120)"
			VERBATIM)
	add_custom_target(TinyGPUCubins DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_cubins.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUNVCubins.h")
	add_dependencies(TinyGPUCubins TinyGPUKernels)
	# The AMD C++ runtime's ahead-of-time HSACOs (TODO.md plan step A1j): every SP and DP variant of BeagleOpenCL_kernels.h
	# compiled for gfx1100 and gfx1201 (TODO.md plan step N13) by tinygpu_amd_compile (tinygrad's compile_hip in C++, through comgr) and embedded by
	# kernels/BeagleTinyGPU_hsaco.S. Without comgr (tinygrad's build: /opt/homebrew/lib/libamd_comgr.dylib, or
	# -DTINYGPU_COMGR=) the plugin is built without them, and refuses an AMD card: nothing compiles at run time since
	# plan step A2l. The HSACOs are outputs too, as the cubins are.
	find_library(TINYGPU_COMGR NAMES amd_comgr HINTS /opt/homebrew/lib)
	if(TINYGPU_COMGR)
		message(STATUS "TinyGPU AMD HSACOs: using comgr at ${TINYGPU_COMGR}")
		add_executable(tinygpu_amd_compile tinygpu_amd_compile.cpp)
		target_include_directories(tinygpu_amd_compile PRIVATE "${CMAKE_SOURCE_DIR}")
		add_dependencies(tinygpu_amd_compile _BeagleOpenCLKernelsFile)
		foreach(a gfx1100 gfx1201)
			foreach(p SP DP)
				foreach(s ${TINYGPU_SP_STATES})
					list(APPEND TINYGPU_HSACOS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tinygpu_hsaco/${p}_${s}_${a}.hsaco")
				endforeach()
			endforeach()
		endforeach()
		add_custom_command(
				OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_hsaco.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUAMDHsaco.h" ${TINYGPU_HSACOS}
				COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_tinygpu_hsaco.sh" "$<TARGET_FILE:tinygpu_amd_compile>" "${TINYGPU_COMGR}"
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
				DEPENDS tinygpu_amd_compile "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleOpenCL_kernels.h" "kernels/make_tinygpu_hsaco.sh"
				COMMENT "Building TinyGPU ahead-of-time AMD HSACOs (comgr: SP and DP x gfx1100, gfx1201)"
				VERBATIM)
		add_custom_target(TinyGPUHsaco DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/BeagleTinyGPU_hsaco.S" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/TinyGPUAMDHsaco.h")
	else()
		message(STATUS "TinyGPU AMD HSACOs: no comgr found; the plugin will refuse AMD cards (no run-time compile)")
	endif()
	# CMake_HIP only defines the HIPKernels target (see the BeagleOpenCL_kernels.h
	# comment above) -- added before CMake_TinyGPU so that target exists
	# by the time CMake_TinyGPU/CMakeLists.txt's add_dependencies needs it.
	add_subdirectory("CMake_HIP")
	add_subdirectory("CMake_TinyGPU")
endif(BUILD_TINYGPU)
