/*
 *
 * Copyright 2009 Phylogenetic Likelihood Working Group
 *
 * This file is part of BEAGLE.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
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
#include <cstdio>
#include "libhmsbeagle/GPU/GPUImplDefs.h"

void checkHostMemory(void* ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Unable to allocate some memory!\n");
        exit(-1);
    }
}

void printfInt(int* ptr,
               int length) {
    fprintf(stderr, "[ %d", ptr[0]);
    int i;
    for (i = 1; i < length; i++)
        fprintf(stderr, " %d", ptr[i]);
    fprintf(stderr, " ]\n");
}
