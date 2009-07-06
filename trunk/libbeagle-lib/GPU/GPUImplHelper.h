/*
 * 
 * @brief GPU implementation helper functions
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __GPUImplHelper__
#define __GPUImplHelper__

#ifdef HAVE_CONFIG_H
#include "libbeagle-lib/config.h"
#endif

#include "libbeagle-lib/GPU/GPUImplDefs.h"

void checkHostMemory(void* ptr);

/**
 * @brief Transposes a square matrix in place
 */
void transposeSquareMatrix(REAL* mat,
                           int size);

void printfVectorD(double* ptr,
                   int length);

void printfVectorF(float* ptr,
                   int length);

void printfVector(REAL* ptr,
                  int length);

void printfInt(int* ptr,
               int length);

#endif // __GPUImplHelper__
