
from distutils import log
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

import sys
import os
import glob
import subprocess

try:
    from Cython.Build import cythonize
except ImportError:
    if "sdist" in sys.argv:
        exit("Error: sdist requires cython module to generate `.c` file.")
    cythonize = False

this_dir = os.path.dirname(__file__)
quickhull_dir = os.path.join(this_dir, "opymize", "quickhull")
source_ext = ".pyx" if cythonize else ".cpp"
sources = [os.path.join(quickhull_dir, "__init__" + source_ext)]
sources += glob.glob(os.path.join(quickhull_dir, "quickhull_src", "*.cpp"))
includes = [
    os.path.join(quickhull_dir, "qhull_src/src/"),
    os.path.join(quickhull_dir, "quickhull_src/"),
]

objects = [ os.path.join(quickhull_dir, "qhull_src/lib/", f) \
            for f in ["libqhullcpp.a", "libqhullstatic_r.a"]]

ext_modules = [
    Extension("opymize.quickhull.__init__",
        sources=sources,
        language="c++",
        include_dirs=includes,
        extra_objects=objects,
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"]),
]

if cythonize:
    ext_modules = cythonize(ext_modules)

class MyBuildExt(build_ext):
    def run(self):
        log.info("building libqhull")
        subprocess.run(["make",
                        "bin-lib","lib/libqhullcpp.a","lib/libqhullstatic_r.a",
                        "CXX_OPTS1=-fPIC -O3 -Isrc/ $(CXX_WARNINGS)",],
                        cwd=os.path.join(quickhull_dir, "qhull_src"))
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)

setup(
    name='opymize',
    version='0.1',
    description='Formulate and solve non-smooth convex optimization problems',
    keywords='convex optimization pdhg',
    url='https://github.com/room-10/Opymize',
    project_urls={ 'Source': 'https://github.com/room-10/Opymize/', },
    author='Thomas Vogt',
    author_email='vogt@mic.uni-luebeck.de',
    packages=find_packages(),
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    package_data={'': ['*.cu']},
    setup_requires=['numpy'],
    install_requires=['numpy','numba','cvxopt','scipy'],
    extras_require={ 'cuda': ['pycuda'], },
    cmdclass={'build_ext': MyBuildExt},
)
