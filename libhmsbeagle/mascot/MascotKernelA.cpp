/*
 * MascotKernelA.cpp -- factory for MASCOT kernel A in one FP mode. Compiled twice:
 *   strict: -ffp-contract=off                  -> mascot::makeKernelAStrict()  (namespace mascot_a_strict)
 *   FMA:    -ffp-contract=fast -DMASCOT_FP_FMA -> mascot::makeKernelAFma()     (namespace mascot_a_fma)
 */
#include "MascotKernelA.hpp"

namespace mascot {

#ifdef MASCOT_FP_FMA
std::unique_ptr<Kernel> makeKernelAFma() { return std::unique_ptr<Kernel>(new mascot_a_fma::KernelA()); }
#else
std::unique_ptr<Kernel> makeKernelAStrict() { return std::unique_ptr<Kernel>(new mascot_a_strict::KernelA()); }
#endif

}  // namespace mascot
