::
:: Windows build script for libhmsbeagle
:: Derived from a similar script for Mauve
:: (c)2008-2009 aaron darling
:: See the file COPYING.LESSER for details
::
::

SET devenv="devenv.exe"
SET vcvars="C:\Progra~2\Microsoft Visual Studio 9.0\VC\vcvarsall.bat"

del errlog.txt

::
:: build 64-bit libs
::
call %vcvars% amd64
@ECHO ON
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Release|x64" /project libhmsbeagle /UseEnv /Out errlog.txt 
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Debug|x64" /project libhmsbeagle /UseEnv /Out errlog.txt 

::
:: build 32-bit libs
::
call %vcvars% x86
@ECHO ON
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Release|Win32" /project libhmsbeagle /UseEnv /Out errlog.txt 
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Debug|Win32" /project libhmsbeagle /UseEnv /Out errlog.txt 

::
:: build installer for 32 and 64-bit
::
start /wait "" %devenv% libhmsbeagle_vc90.sln /build "Release|Win32" /project "hmsbeagle library" /UseEnv /Out errlog.txt 

