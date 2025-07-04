from setuptools import setup, find_packages

setup(
    name='IHSetJaramillo21a',
    version='1.3.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'scipy',
        'pandas',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Jaramillo et al. (2021a)',
    url='https://github.com/defreitasL/IHSetJaramillo21a',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)