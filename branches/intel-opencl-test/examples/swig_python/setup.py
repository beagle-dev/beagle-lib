from distutils.core import setup, Extension
import commands

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw

beagle_module = Extension("_beagle",sources=['beagle_wrap.c'],**pkgconfig('hmsbeagle-1'))

setup(name='beagle',
    version='0.1',
    author="Simon Frost",
    description="""BEAGLE module""",
    ext_modules = [beagle_module],
    py_modules = ["beagle"],
    )

