from setuptools import find_packages, setup

setup(
    name='xaa',
    packages=find_packages(where='src', include='xaa*'),
    package_dir={"": "src"},
    version='0.1.0',
    description='X-Ray absorption analysis tools',
    author='Benjamin Froelich',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'lmfit', 'PyQt5']
)