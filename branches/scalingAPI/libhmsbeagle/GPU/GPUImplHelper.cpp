/*
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include <cstdlib>
#include <iostream>
#include <cmath>
#include "libhmsbeagle/GPU/GPUImplDefs.h"

void checkHostMemory(void* ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Unable to allocate some memory!\n");
        exit(-1);
    }
}

void transposeSquareMatrix(REAL* mat,
                           int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            REAL tmp = mat[i * size + j];
            mat[i * size + j] = mat[j * size + i];
            mat[j * size + i] = tmp;
        }
    }
}

void printfVectorD(double* ptr,
                   int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

void printfVectorF(float* ptr,
                   int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
    for(i = 0; i < length; i++) {
        if(ptr[i] != ptr[i]) {
            fprintf(stderr, "NaN found!\n");
            exit(0);
        }
    }
}

void printfVector(REAL* ptr,
                  int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

void printfInt(int* ptr,
               int length) {
    fprintf(stderr, "[ %d", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %d", ptr[i]);
    fprintf(stderr, " ]\n");
}
