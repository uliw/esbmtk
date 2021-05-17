""" Setup file for esbmtk

sphinx directives:

.  automodule:: package.module

"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="esbmtk",  # Replace with your own username
    version="0.5.1.4",
    author="Ulrich G. Wortmann",
    license="GPL-3.0-or-later",
    author_email="uli.wortmann@utoronto.ca",
    description="An Earth Sciences Box Modeling Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uliw/esbmtk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    ],
    python_requires=">=3.7",
    install_requires=[
        "nptyping",
        "typing",
        "numpy",
        "pandas",
        "matplotlib",
        "pint",
        "scipy",
        "numba",
    ],
)
