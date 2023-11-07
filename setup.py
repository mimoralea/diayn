
from setuptools import setup, find_packages

setup(
    name='diayn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'diayn=diayn.__main__:main'
        ]
    }
)
