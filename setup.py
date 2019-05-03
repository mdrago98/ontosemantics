import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioengine",
    version="0.0.1",
    author="Matthew Drago",
    author_email="matthew.drago.16@um.edu.mt",
    description="A library for processing pubmed abstracts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdrago98/bioengine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)