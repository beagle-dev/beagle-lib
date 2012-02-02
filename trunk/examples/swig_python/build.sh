export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/lib/pkgconfig:$PKG_CONFIG_PATH

swig -python beagle.i 
gcc -fPIC -c beagle_wrap.c -I. -I/usr/include/python2.7 `pkg-config --cflags --libs hmsbeagle-1`
gcc -shared beagle_wrap.o -o _beagle.so `pkg-config --cflags --libs hmsbeagle-1`

