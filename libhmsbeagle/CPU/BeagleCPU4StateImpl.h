/*
 *  BeagleCPUImpl.h
 *  BEAGLE
 *
 * @author Andrew Rambaut
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __BeagleCPU4StateImpl__
#define __BeagleCPU4StateImpl__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/CPU/BeagleCPUImpl.h"

#include <vector>

namespace beagle {
namespace cpu {

class BeagleCPU4StateImpl : public BeagleCPUImpl {
        
public:
    virtual ~BeagleCPU4StateImpl();
    
private:
    virtual void calcStatesStates(double * destP,
                          const int * child0States,
                          const double *child0TransMat,
                          const int * child1States,
                          const double *child1TransMat);
    
    virtual void calcStatesPartials(double * destP,
                            const int * child0States,
                            const double *child0TransMat,
                            const double * child1Partials,
                            const double *child1TransMat);
    
    virtual void calcPartialsPartials(double * destP,
                              const double * child0States,
                              const double *child0TransMat,
                              const double * child1Partials,
                              const double *child1TransMat);
};

class BeagleCPU4StateImplFactory : public BeagleImplFactory {
public:
    virtual BeagleImpl* createImpl(int tipCount,
                                   int partialsBufferCount,
                                   int compactBufferCount,
                                   int stateCount,
                                   int patternCount,
                                   int eigenBufferCount,
                                   int matrixBufferCount,
                                   int categoryCount,
                                   int scaleBufferCount);
    
    virtual const char* getName();
};

}	// namespace cpu
}	// namespace beagle

#endif // __BeagleCPU4StateImpl__
