/*	linalg.h
|
|	Prototypes for matrix-inversion and eigensystem functions
|
|	Copyright (c) 1998 by David L. Swofford, Smithsonian Institution.
|	All rights reserved.
|
|	NOTE: if ANSI function prototypes are not supported, define NO_PROTOTYPES
|		  before including this file.
*/

#define RC_COMPLEX_EVAL 2	/* code that complex eigenvalue obtained */

extern int  InvertMatrix (double **a, int n, double *col, int *indx, double **a_inv);
extern int  LUDecompose (double **a, int n, double *vv, int *indx, double *pd);
int  EigenRealGeneral (int n, double **a, double *v, double *vi, double **u, int *iwork, double *work);


template<typename T> T **New2DArray(unsigned f , unsigned s)
{
	T **temp;
	temp = new T *[f];
	*temp = new T [f * s];
	for (unsigned fIt = 1 ; fIt < f ; fIt ++)
		temp[fIt] = temp[fIt -1] +  s ;
	return temp;
}

/*--------------------------------------------------------------------------------------------------------------------------
 | Delete a 2 Dimensional Array New2DArray
 */
template<typename T> inline void Delete2DArray	(T **temp)
{
	if (temp)
    {
		if (*temp)
			delete [] * temp;
		delete [] temp;
    }
}