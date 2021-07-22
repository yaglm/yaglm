from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

# time, abc, numbers, copy, textwrap
# os, json, argparase, re

install_requires = ['numpy',
                    'pandas',
                    'scikit-learn',
                    'scipy'
                    ]


setup(name='ya_glm',
      version='0.1.3',
      description='A flexible package for fitting penalized generalized linear models.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
