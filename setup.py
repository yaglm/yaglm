from packaging.version import parse
from setuptools import setup, find_packages

import os

version = None
with open(os.path.join('yaglm', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = parse(line.split('=')[1].strip().strip('\''))
            break
if version is None:
    raise RuntimeError('Could not determine version')


install_requires = ['numpy',
                    'pandas',
                    'scikit-learn',
                    'scipy'
                    ]


setup(name='yaglm',
      version=version,
      description='A python package for penalized generalized linear models that supports fitting and model selection for structured, adaptive and non-convex penalties.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
