/*
 * Precision.h
 *
 *  Created on: Oct 13, 2009
 *      Author: msuchard
 */

#ifndef PRECISION_H_
#define PRECISION_H_

#define DOUBLE_PRECISION (sizeof(REALTYPE) == 8)

template<typename T, typename F> 
inline void beagleMemCpy( T* to, F* from, unsigned int length )
{
	for(unsigned int m=0; m<length; m++)
		to[m]=(T)from[m];
};

template<typename F> 
inline void beagleMemCpy( F* to, const F* from, unsigned int length )
{
	memcpy( to, from, length*sizeof(F) );
}

/*#define MEMCNV(to, from, length, toType)    { \
                                                int m; \
                                                for(m = 0; m < length; m++) { \
                                                    to[m] = (toType) from[m]; \
                                                } \
                                            }
*/

#endif /* PRECISION_H_ */
