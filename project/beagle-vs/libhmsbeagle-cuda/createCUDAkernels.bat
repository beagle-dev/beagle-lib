::
:: windows script to create cuda files for each state count
:: from the generic state count file
:: @author Aaron Darling
::
:: TODO: make this pretty with a loop over state counts
::
cd ..\..\..\libhmsbeagle\GPU\kernels

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels16.cu
echo #define STATE_COUNT 16 >> kernels16.cu
type kernelsX.cu >> kernels16.cu

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels32.cu
echo #define STATE_COUNT 32 >> kernels32.cu
type kernelsX.cu >> kernels32.cu

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels48.cu
echo #define STATE_COUNT 48 >> kernels48.cu
type kernelsX.cu >> kernels48.cu

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels64.cu
echo #define STATE_COUNT 64 >> kernels64.cu
type kernelsX.cu >> kernels64.cu

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels128.cu
echo #define STATE_COUNT 128 >> kernels128.cu
type kernelsX.cu >> kernels128.cu

echo // DO NOT EDIT -- autogenerated file -- edit kernelsX.cu instead > kernels192.cu
echo #define STATE_COUNT 192 >> kernels192.cu
type kernelsX.cu >> kernels192.cu