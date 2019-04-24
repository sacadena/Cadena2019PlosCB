#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='cnn_sys_ident',
    version='0.0.0',
    description='Deep learning models for V1 neural system identification',
    author='Santiago Cadena',
    author_email='santiago.cadena@uni-tuebingen.de',
    url='https://github.com/sacadena/Cadena2019PlosCB',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'scipy', 'tensorflow', 'scikit-image', 'datajoint'],
)