#!/bin/sh
mkdir -p config
case `uname` in Darwin*) glibtoolize ;; *) libtoolize ;; esac
autoreconf --force --install -I config -I m4
