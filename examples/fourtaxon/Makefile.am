check_PROGRAMS = fourtaxon
fourtaxon_SOURCES = fourtaxon.cpp fourtaxon.hpp
fourtaxon_LDADD = $(top_builddir)/$(GENERIC_LIBRARY_NAME)/libhmsbeagle.la @CUDA_LIBS@

check_SCRIPTS = fourtaxonrun.sh
fourtaxonrun.sh:
	echo './fourtaxon --niters 4 --filename $(srcdir)/fourtaxon.dat' > fourtaxonrun.sh
	chmod +x fourtaxonrun.sh

clean-local:
	rm -f fourtaxonrun.sh 
	rm -f check_lnL_using_paup.nex

TESTS = fourtaxonrun.sh
INCLUDES = -I$(top_builddir) -I$(top_srcdir)

EXTRA_DIST=fourtaxon.dat

