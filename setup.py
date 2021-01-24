
from setuptools import setup

with open("README.md", 'rb') as fh:
    long_description = fh.read().decode('utf-8')

setup(name='DaiToolkit',
      version='0.0.1',
      description='Finance tools for analytics/trading',
      long_description=long_description,
      author='Guochen Dai',
      author_email='gd1023@nyu.edu')