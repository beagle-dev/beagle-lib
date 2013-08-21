/*
 * Precision.h
 *
 *  Created on: Oct 13, 2009
 *      Author: msuchard
 */

#ifndef GPU_PRECISION_H_
#define GPU_PRECISION_H_

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

template<typename F>
inline const F* beagleCastIfNecessary(const F* from, F* cache,
		unsigned int length) {
	return from;
}

template<typename T, typename F>
inline const T* beagleCastIfNecessary(const F* from, T* cache,
		unsigned int length) {
	beagleMemCpy(cache, from, length);
	return cache;
}
//
//template<typename F>
//inline void beagleCopyFromDeviceAndCastIfNecessary(GPUInterface* gpu, F* to, const F* from, F* cache,
//		unsigned int length) {
//	gpu->MemcpyDeviceToHost(to, from, sizeof(F) * kPatternCount);
//}
//
//template<typename T, typename F>
//inline void beagleCopyFromDeviceAndCastIfNecessary(GPUInterface* gpu, T* to, const F* from, F* cache,
//		unsigned int length) {
//	gpu->MemcpyDeviceToHost(cache, from, sizeof(F) * kPatternCount);
//	beagleMemCpy(to, cache, kPatternCount);
//}


//#ifdef DOUBLE_PRECISION
//    gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, sizeof(Real) * kPatternCount);
//#else
//    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(Real) * kPatternCount);
//    MEMCNV(outLogLikelihoods, hLogLikelihoodsCache, kPatternCount, double);
//#endif



#endif /* GPU_PRECISION_H_ */
