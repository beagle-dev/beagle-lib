/*
 * Precision.h
 *
 *  Created on: Oct 13, 2009
 *      Author: msuchard
 */

#ifndef PRECISION_H_
#define PRECISION_H_

#define DOUBLE_PRECISION (sizeof(REALTYPE) == 8)

#define MEMCNV(to, from, length, toType)    { \
                                                int m; \
                                                for(m = 0; m < length; m++) { \
                                                    to[m] = (toType) from[m]; \
                                                } \
                                            }

#endif /* PRECISION_H_ */
