from setuptools import setup, find_packages


setup(
    name="song2vec",
    version="0.1.0",
    description="python package for training vector embedding models",
    author="Bastian Lipka",
    author_email="lipka.bastian@gmail.com",
    packages=find_packages(),  # Automatically find and include package directories
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

# still need some fix here to get the .pyd file setup

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("song2vec/submodule/song2vec.pyx"),
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--build-lib', 'build']
)