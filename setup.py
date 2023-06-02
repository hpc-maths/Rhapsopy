from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rhapsopy",
    version="0.0.2",
    author="Laurent François",
    author_email="laurent.francois@polytechnique.edu",
    description="Really High-order AdaPtive code coupling for SImulation Orchestration in PYthon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TODO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy'],
)

# run "python setup.py develop" once !
