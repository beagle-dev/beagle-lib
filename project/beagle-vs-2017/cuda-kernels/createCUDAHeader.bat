::
:: windows script to create a single header with compiled CUDA 
:: @author Aaron Darling
:: @author Daniel Ayres
:: 

set OUTFILE="..\..\..\libhmsbeagle\GPU\kernels\BeagleCUDA_kernels.h"

echo // auto-generated header file with CUDA kernels code > %OUTFILE%

echo #ifndef __BeagleCUDA_kernels__ >> %OUTFILE%
echo #define __BeagleCUDA_kernels__ >> %OUTFILE%

FOR %%G IN (4 16 32 48 64 80 128 192) DO (

..\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_SP_%%G ..\cuda-kernels\data\cuda\ptx\kernels%%G.ptx >> %OUTFILE%
..\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_DP_%%G ..\cuda-kernels\data\cuda\ptx\kernels_dp_%%G.ptx >> %OUTFILE%

)

echo #endif 	// __BeagleCUDA_kernels__ >> %OUTFILE%
