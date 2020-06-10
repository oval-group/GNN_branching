import sys
import glob

from os import path
from setuptools import setup, find_packages
from setuptools.extension import Extension

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='GNN-branching',
    version='0.0.1',
    description='Neural Network Branching for Neural Network Verification',
    author='Jingyue Lu',
    author_email='jingyue.lu@spc.ox.ac.uk',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'scipy'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)
