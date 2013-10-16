::
:: windows script to create a single header with compiled CUDA 
:: kernels defined as variables
:: @author Aaron Darling
::
:: TODO: make this pretty with a loop over state counts
::


set OUTFILE="..\..\..\libhmsbeagle\GPU\kernels\BeagleCUDA_kernels.h"
if exist %OUTFILE% del %OUTFILE%

echo #ifndef __BeagleCUDA_kernels__ >> %OUTFILE%
echo #define __BeagleCUDA_kernels__ >> %OUTFILE%

FOR %%G IN (4 16 32 48 64 80 128 192) DO (

bin2c.exe -st -n KERNELS_STRING_SP_%%G data\kernels%%G.ptx >> %OUTFILE%
bin2c.exe -st -n KERNELS_STRING_DP_%%G data\kernels_dp_%%G.ptx >> %OUTFILE%

)

echo #endif 	// __BeagleCUDA_kernels__ >> %OUTFILE%

