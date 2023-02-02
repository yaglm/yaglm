from setuptools import setup, find_packages

from yaglm import __version__ as version


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
