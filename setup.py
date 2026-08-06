from setuptools import setup

setup(
    name='ResidualNetworks',
    version='0.1.0',
    description='Residual network examples for optical communication constellations and NSGA2 optimization.',
    packages=['ResidualNetworks'],
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        'ResidualNetworks': ['data/*.csv'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'pymoo',
    ],
    python_requires='>=3.12',
    zip_safe=False,
)
