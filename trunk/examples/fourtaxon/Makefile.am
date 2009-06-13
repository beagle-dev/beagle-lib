check_PROGRAMS = fourtaxon
fourtaxon_SOURCES = fourtaxon.cpp fourtaxon.hpp
fourtaxon_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libbeagle-lib.la

check_SCRIPTS = fourtaxonrun.sh
fourtaxonrun.sh:
	echo './fourtaxon 1000 fourtaxon.dat' > fourtaxonrun.sh
	chmod +x fourtaxonrun.sh

TESTS = fourtaxonrun.sh

EXTRA_DIST=fourtaxon.dat

