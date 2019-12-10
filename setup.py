from setuptools import setup, find_packages

from oggdo.version import __version__

setup(
    name="fragile",
    version=__version__,
    author="Ceshine Lee",
    author_email="ceshine@ceshine.net",
    description="",
    license="GLWT(Good Luck With That)",
    url="",
    packages=find_packages(exclude=['scripts']),
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=""
)
