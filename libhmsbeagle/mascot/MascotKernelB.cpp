/*
 * MascotKernelB.cpp -- factory for MASCOT kernel B (MascotKernelB.hpp). This file is compiled twice, both times with
 * -ffp-contract=off:
 *  - strict: defines makeKernelBStrict(), bitwise with Java on gradients and ancestral scores;
 *  - with -DMASCOT_FP_FMA: defines makeKernelBFma(), which fuses exactly the multiply-adds the kernel writes with
 *    FMAV / FMAS (MascotFma.hpp).
 * The kernel code lives in a namespace that differs per mode (mascot_b_strict / mascot_b_fma), so the linker never
 * merges inline functions compiled for different FP modes.
 */
#include "MascotKernelB.hpp"

namespace mascot {

#ifdef MASCOT_FP_FMA
std::unique_ptr<Kernel> makeKernelBFma() { return std::unique_ptr<Kernel>(new MASCOT_B_NS::KernelB()); }
#else
std::unique_ptr<Kernel> makeKernelBStrict() { return std::unique_ptr<Kernel>(new MASCOT_B_NS::KernelB()); }
#endif

} // namespace mascot
