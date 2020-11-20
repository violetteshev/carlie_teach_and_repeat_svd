#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

import os

d = generate_distutils_setup(
  packages=['teach_repeat'], 
  package_dir={'': 'src'}
)

setup(**d)
