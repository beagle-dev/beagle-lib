/*
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
