/*
 * @brief GPU implementation helper functions
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * BEAGLE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * BEAGLE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAGLE.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __GPUImplHelper__
#define __GPUImplHelper__

#ifdef HAVE_CONFIG_H
#include "libhmsbeagle/config.h"
#endif

#include "libhmsbeagle/GPU/GPUImplDefs.h"

void checkHostMemory(void* ptr);

/**
 * @brief Transposes a square matrix in place
 */
template<typename Real>
void transposeSquareMatrix(Real* mat,
                           int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            Real tmp = mat[i * size + j];
            mat[i * size + j] = mat[j * size + i];
            mat[j * size + i] = tmp;
        }
    }
}

template<typename Real>
void printfVector(Real* ptr,
                  int length) {
    fprintf(stderr, "[ %1.5e", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %1.5e", ptr[i]);
    fprintf(stderr, " ]\n");
}

void printfInt(int* ptr,
               int length);

#endif // __GPUImplHelper__
