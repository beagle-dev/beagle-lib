set OUTFILE="..\..\libhmsbeagle\GPU\kernels\BeagleCUDA_kernels.h"
if exist %OUTFILE% del %OUTFILE%

cudaBuildHelper.exe data\kernels4.ptx KERNELS_STRING_4 %OUTFILE%
cudaBuildHelper.exe data\kernels4ls.ptx KERNELS_STRING_LS_4 %OUTFILE%
cudaBuildHelper.exe data\kernels32.ptx KERNELS_STRING_32 %OUTFILE%
cudaBuildHelper.exe data\kernels32ls.ptx KERNELS_STRING_LS_32 %OUTFILE%
cudaBuildHelper.exe data\kernels48.ptx KERNELS_STRING_48 %OUTFILE%
cudaBuildHelper.exe data\kernels48ls.ptx KERNELS_STRING_LS_48 %OUTFILE%
cudaBuildHelper.exe data\kernels64.ptx KERNELS_STRING_64 %OUTFILE%
cudaBuildHelper.exe data\kernels64ls.ptx KERNELS_STRING_LS_64 %OUTFILE%
