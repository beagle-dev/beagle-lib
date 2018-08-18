IF DEFINED JDK_HOME (GOTO END) ELSE (GOTO FIND_JAVA_HOME)
:FIND_JAVA_HOME
echo Hunting for Java
FOR /F "skip=2 tokens=2*" %%A IN ('REG QUERY "HKLM\SOFTWARE\JavaSoft\Java Development Kit" /v CurrentVersion') DO set CurVer=%%B
FOR /F "skip=2 tokens=2*" %%A IN ('REG QUERY "HKLM\SOFTWARE\JavaSoft\Java Development Kit\%CurVer%"  /v JavaHome') DO set JDK_HOME=%%B
:END

echo #pragma include_alias(^<jni.h^>, ^<%JDK_HOME%\include\jni.h^>) > ..\..\libhmsbeagle\JNI\winjni.h
echo #pragma include_alias("jni_md.h", "%JDK_HOME%\include\win32\jni_md.h") >> ..\..\libhmsbeagle\JNI\winjni.h