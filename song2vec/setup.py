# setup.py

from setuptools import setup, Extension, find_packages
import os
import sys

# Determine the compiler flags based on the platform
extra_compile_args = []
if os.name == "nt":
    # Windows uses MSVC
    extra_compile_args = ["/O2"]  # Optimize for speed
else:
    # Unix-like systems use GCC/Clang
    extra_compile_args = ["-O3"]

# Define the C extension module
c_extension = Extension(
    "song2vec.submodule.c_extension.song2vec_c",  # Full module name
    sources=[os.path.join("song2vec", "submodule", "c_extension", "song2vec_c.c")],
    extra_compile_args=extra_compile_args,
)

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup configuration
setup(
    name="song2vec",
    version="1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Song2Vec Python Package with C extension for optimized performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/song2vec",  # Update with your repo
    packages=find_packages(),
    ext_modules=[c_extension],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
