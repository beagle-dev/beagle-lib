#!/bin/sh
#
# The following is for configuring a version of Beagle for embedding into an OSX application bundle:
#
# Both 32-bit and 64-bit libraries are built
#
# Remember that "configure" script doesn't exist until autogen.sh has been run.
#

./configure-macos-universal.sh
./make-macos-universal.sh
