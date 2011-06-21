::
:: Windows build script for libhmsbeagle
:: Derived from a similar script for Mauve
:: (c)2008-2009 aaron darling
:: See the file COPYING.LESSER for details
::
:: Usage: set devenv and vcvars and ant to values appropriate for your system
::

SET devenv="devenv.exe"
SET vcvars="C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\vcvarsall.bat"

del errlog.txt

::
:: Step 1: Find Java and build the beagle java API library
::
IF DEFINED JDK_HOME (GOTO END) ELSE (GOTO FIND_JAVA_HOME)
:FIND_JAVA_HOME
echo Hunting for Java
FOR /F "skip=2 tokens=2*" %%A IN ('REG QUERY "HKLM\SOFTWARE\JavaSoft\Java Development Kit" /v CurrentVersion') DO set CurVer=%%B
FOR /F "skip=2 tokens=2*" %%A IN ('REG QUERY "HKLM\SOFTWARE\JavaSoft\Java Development Kit\%CurVer%"  /v JavaHome') DO set JDK_HOME=%%B
:END

echo #pragma include_alias(^<jni.h^>, ^<%JDK_HOME%\include\jni.h^>) > ..\..\libhmsbeagle\JNI\winjni.h
echo #pragma include_alias("jni_md.h", "%JDK_HOME%\include\win32\jni_md.h") >> ..\..\libhmsbeagle\JNI\winjni.h

::
:: Step 2: build 64-bit libs
::
call %vcvars% amd64
@ECHO ON
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Release|x64" /UseEnv /Out errlog.txt 
:: start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Debug|x64" /UseEnv /Out errlog.txt 

::
:: Step 3: build 32-bit libs
::
call %vcvars% x86
@ECHO ON
start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Release|Win32" /UseEnv /Out errlog.txt 
:: start /wait "" %devenv% libhmsbeagle_vc90.sln /rebuild "Debug|Win32" /UseEnv /Out errlog.txt 


::
:: Step 4: build installer for 32 and 64-bit
::
start /wait "" %devenv% libhmsbeagle_vc90.sln /build "Release|Win32" /project "hmsbeagle library" /UseEnv /Out errlog.txt 

