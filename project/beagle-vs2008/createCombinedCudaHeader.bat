set OUTFILE="..\..\libhmsbeagle\GPU\kernels\BeagleCUDA_kernels.h"
if exist %OUTFILE% del %OUTFILE%

bin2c.exe -n KERNELS_STRING_4 data\kernels4.ptx > %OUTFILE%
bin2c.exe -n KERNELS_STRING_LS_4 data\kernels4ls.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_32 data\kernels32.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_LS_32 data\kernels32ls.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_48 data\kernels48.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_LS_48 data\kernels48ls.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_64 data\kernels64.ptx >> %OUTFILE%
bin2c.exe -n KERNELS_STRING_LS_64 data\kernels64ls.ptx >> %OUTFILE%
