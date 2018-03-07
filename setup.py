
from setuptools import setup, find_packages

setup(
    name='opymize',
    version='0.1',
    description='Formulate and solve non-smooth convex optimization problems',
    urls='https://github.com/room-10/Opymize',
    author='Thomas Vogt',
    author_email='vogt@mic.uni-luebeck.de',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='convex optimization pdhg',
    packages=find_packages(),
    install_requires=['numpy','numba'],
    extras_require={ 'cuda': ['pycuda'], },
    project_urls={ 'Source': 'https://github.com/room-10/Opymize/', },
)
