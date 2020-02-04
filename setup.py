from setuptools import find_packages, setup

setup(
    name='mlmodels',
    packages=find_packages(),
    version='0.1.1',
    description='Base model and transformer classes.',
    author='Jes Ravnbøl',
    license='',
    install_requires=[
        'gorilla',
        'mlflow',
        'simplejson',
        'scikit-learn',
        'pandas',
        'jinja2',
        'PyYAML',
        'numpy'
    ],
)
