import os

import setuptools
from pip._internal.req import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]

setuptools.setup(
    name="bioengine",
    version="0.0.1",
    author="Matthew Drago",
    author_email="matthew.drago.16@um.edu.mt",
    description="A library for processing pubmed abstracts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdrago98/bioengine",
    packages=setuptools.find_packages(exclude=['resources*', 'docs*', 'tests*', 'evaluation*']),
    install_requires=['cython', 'numpy', 'py2neo', 'spacy==2.0.18'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
