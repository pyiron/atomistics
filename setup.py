"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name='atomistics',
    version=versioneer.get_version(),
    description='atomistics - materials science workgflows to calculate material properties',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/atomistics',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11'
                ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'ase==3.22.1',
        'numpy==1.23.5',
        'scipy==1.11.1',
        'spglib==2.0.2',
    ],
    cmdclass=versioneer.get_cmdclass(),
)
