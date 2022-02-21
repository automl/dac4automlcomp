from setuptools import setup

packages = [
    "dac4automlcomp",
]

package_data = {"": ["*"]}

AUTHORS = (
    ", ".join(
        [
            "Raghu Rajan",
        ]
    ),
)

AUTHOR_EMAIL = "rajanr@cs.uni-freiburg.de"

# TODO
LICENSE = "Apache License, Version 2.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dac4automlcomp",
    py_modules=["dac4automlcomp.run_experiments"],
    version="0.0.1",
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    description="A python package for the DAC4AutoML competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    # url="https://github.com/automl/", #TODO
    project_urls={
        "Bug Tracker": "https://github.com/automl/#TODO/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",  # TODO
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=packages,
    # package_dir={"": "src"},
    python_requires=">=3.8",
    setup_requires=[""],
    install_requires=["", ""],
    extras_require={},
)
