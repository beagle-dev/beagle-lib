#!/bin/sh
# configure universal macOS build
#
# Remember that "configure" script doesn't exist until autogen.sh has been run.

BEAGLE_ROOT=`pwd`/../..
rm -rf build32 build64 install32 install64 installFat

mkdir build32
cd build32
$BEAGLE_ROOT/configure "CXXFLAGS=-m32" --prefix=`pwd`/install --without-cuda
cd ..

mkdir build64
cd build64
$BEAGLE_ROOT/configure --prefix=`pwd`/install --with-cuda=/usr/local/cuda
cd ..


