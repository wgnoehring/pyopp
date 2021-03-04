import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ovito_scripts", 
    version="0.1.0",
    author="Wolfram Georg Nöhring",
    author_email="wolfram.noehring@imtek.uni-freiburg.de",
    description="Scripts for postprocessing using Ovito",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=[
        "ovito_scripts", 
        "ovito_scripts.displacements", 
        "ovito_scripts.util", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
