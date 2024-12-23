# -*- coding: utf-8 -*-
"""
# This file is part of pyGNDiv.
# Copyright 2023 Javier Pacheco-Labrador and contributors listed in the
# README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages

setup(name='pyGNDiv',
      version='0.0.1',
      description='Global Normalization Functional Diversity Metrics',
      long_description='Global Normalization of Functional Diversity Metrics in Remote Sensing',
      url='https://github.com/JavierPachecoLabrador/pyGNDiv.git',
      author='Javier Pacheco-Labrador',
      author_email='javier.pacheco@csic.es',
      license='GPL',
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Biodiveristy',
                   'Topic :: Scientific/Remote Sensing',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7'
                   'Programming Language :: Python :: 3.8'
                   'Programming Language :: Python :: 3.9'
                   'Programming Language :: Python :: 3.10'
                   'Programming Language :: Python :: 3.11'
                   'Programming Language :: Python :: 3.12'],
      keywords=['Remote Sensing', 'Plant Functional Diversity',
                'Rao Quadratic Entropy', 'Functional Richness',
                'dissimilarity', 'Normalization', 'Global'],
      packages=find_packages(),
      install_requires=['numpy', 'scikit-learn', 'scipy', 'more-itertools',
                        'python-math', 'wpca', 'numba'],
      zip_safe=True)
