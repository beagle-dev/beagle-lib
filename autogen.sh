#!/bin/sh
mkdir -p config
libtoolize
autoreconf --force --install -I config -I m4
