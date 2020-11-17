from setuptools import setup

setup(
    name='odc_algo',

    use_scm_version={"root": "../..", "relative_to": __file__},
    setup_requires=['setuptools_scm'],

    author='Open Data Cube',
    author_email='',
    maintainer='Open Data Cube',

    description='Miscellaneous Algorithmic helper methods',
    long_description='',

    license='Apache License 2.0',

    tests_require=['pytest'],
    install_requires=[
        'affine',
        'numexpr',
        'dask',
        'distributed',
        'xarray',
        'numpy',
        'toolz',
        'hdstats>=0.1.7.post5',
        'odc-index',
        'datacube',
        'scikit-image',
        'dask_image'
    ],

    packages=['odc.algo'],
    zip_safe=False,
)
