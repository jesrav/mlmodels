from setuptools import find_packages, setup

setup(
    name='mlmodels',
    packages=find_packages(),
    version='0.1.0',
    description='Base model and transformer classes.',
    author='Jes Ravnbøl',
    license='',
    install_requires=['scikit-learn', 'pandas', 'numpy', 'mlflow', 'apispec', 'jinja2', 'PyYAML'],
)
