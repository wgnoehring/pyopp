import setuptools
import versioneer

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyopp", 
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Wolfram Georg Nöhring",
    author_email="wolfram.noehring@imtek.uni-freiburg.de",
    description="Scripts for postprocessing using Ovito",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=[
        "pyopp", 
        "pyopp.displacements", 
        "pyopp.util", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
