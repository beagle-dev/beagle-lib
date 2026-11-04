::
:: windows script to create BeagleOpenCLSpectral_kernels.h, the counterpart of make_opencl_spectral_kernels.sh:
:: per state count, one string with the kernel source and the derivative and spectral kernels
:: (kernels4Derivatives.cu and kernelsSpectralIfDef4.cu for 4 states, kernelsXDerivatives.cu and
:: kernelsSpectralIfDef.cu for the others)
:: The temporary files are spectral_kernels*.cl, apart from createOpenCLHeader.bat's kernels*.cl
::
cd ..\..\..\libhmsbeagle\GPU\kernels
echo #define STATE_COUNT 4 > spectral_kernels4.cl
type ..\GPUImplDefs.h >> spectral_kernels4.cl
type kernelsAll.cu >> spectral_kernels4.cl
type kernels4.cu >> spectral_kernels4.cl
type kernels4Derivatives.cu >> spectral_kernels4.cl
type kernelsSpectralIfDef4.cu >> spectral_kernels4.cl

FOR %%G IN (16 32 48 64 80 128 192 256) DO (

echo #define STATE_COUNT %%G> spectral_kernels%%G.cl
type ..\GPUImplDefs.h >> spectral_kernels%%G.cl
type kernelsAll.cu >> spectral_kernels%%G.cl
type kernelsX.cu >> spectral_kernels%%G.cl
type kernelsXDerivatives.cu >> spectral_kernels%%G.cl
type kernelsSpectralIfDef.cu >> spectral_kernels%%G.cl

)

echo #define STATE_COUNT 4 > spectral_kernels_dp_4.cl
echo #define DOUBLE_PRECISION>> spectral_kernels_dp_4.cl
type ..\GPUImplDefs.h >> spectral_kernels_dp_4.cl
type kernelsAll.cu >> spectral_kernels_dp_4.cl
type kernels4.cu >> spectral_kernels_dp_4.cl
type kernels4Derivatives.cu >> spectral_kernels_dp_4.cl
type kernelsSpectralIfDef4.cu >> spectral_kernels_dp_4.cl

FOR %%G IN (16 32 48 64 80 128 192 256) DO (

echo #define STATE_COUNT %%G> spectral_kernels_dp_%%G.cl
echo #define DOUBLE_PRECISION>> spectral_kernels_dp_%%G.cl
type ..\GPUImplDefs.h >> spectral_kernels_dp_%%G.cl
type kernelsAll.cu >> spectral_kernels_dp_%%G.cl
type kernelsX.cu >> spectral_kernels_dp_%%G.cl
type kernelsXDerivatives.cu >> spectral_kernels_dp_%%G.cl
type kernelsSpectralIfDef.cu >> spectral_kernels_dp_%%G.cl

)

set OUTFILE="BeagleOpenCLSpectral_kernels.h"

echo // auto-generated header file with OpenCL spectral kernels code > %OUTFILE%

echo #ifndef __BeagleOpenCLSpectral_kernels__ >> %OUTFILE%
echo #define __BeagleOpenCLSpectral_kernels__ >> %OUTFILE%

FOR %%G IN (4 16 32 48 64 80 128 192 256) DO (

..\..\..\project\beagle-vs-2019-wix\bin2c.exe -p 0 -st -n KERNELS_STRING_SP_%%G spectral_kernels%%G.cl >> %OUTFILE%
..\..\..\project\beagle-vs-2019-wix\bin2c.exe -p 0 -st -n KERNELS_STRING_DP_%%G spectral_kernels_dp_%%G.cl >> %OUTFILE%

del spectral_kernels%%G.cl
del spectral_kernels_dp_%%G.cl

)

echo #endif 	// __BeagleOpenCLSpectral_kernels__ >> %OUTFILE%
