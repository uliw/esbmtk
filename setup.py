import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="esbmtk",  # Replace with your own username
    version="0.2.1.1",
    author="Ulrich G. Wortmann",
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
    ],
    python_requires=">=3.6",
    install_requires=[
        "nptyping",
        "typing",
        "numpy",
        "pandas",
        "matplotlib",
        "pint",
    ],
)
