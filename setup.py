from setuptools import setup, find_packages

with open('requirements.txt') as f:
        requirements = f.read().splitlines()

requirements = [
    "graspologic",
    # "proglearn",
    "sklearn",
    "numpy",
    "matplotlib",
    "seaborn",
    # 'tasksim @ git+git://github.com/neurodata/task-similarity.git#egg=tasksim'
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="hierarchical",
    version="0.0.1",
    author="Hayden Helm",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/hierarchical/",
#    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
#        "License :: OSI Approved :: MIT License",
#        "Programming Language :: Python :: 3",
#        "Programming Language :: Python :: 3.6",
#        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
#    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
#    include_package_data=True
)
