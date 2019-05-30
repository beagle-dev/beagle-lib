#!/bin/sh
# make universal macOS build

cd build64
make install
cd ..

cd build32
make install
cd ..

cp -R build64/install installFat

lipo -create build32/install/lib/libhmsbeagle.1.dylib build64/install/lib/libhmsbeagle.1.dylib -output installFat/lib/libhmsbeagle.1.dylib
lipo -create build32/install/lib/libhmsbeagle-cpu.31.so build64/install/lib/libhmsbeagle-cpu.31.so -output installFat/lib/libhmsbeagle-cpu.31.so
lipo -create build32/install/lib/libhmsbeagle-cpu-sse.31.so build64/install/lib/libhmsbeagle-cpu-sse.31.so -output installFat/lib/libhmsbeagle-cpu-sse.31.so
lipo -create build32/install/lib/libhmsbeagle-jni.jnilib build64/install/lib/libhmsbeagle-jni.jnilib -output installFat/lib/libhmsbeagle-jni.jnilib
