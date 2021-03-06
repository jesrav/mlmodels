from setuptools import find_packages, setup

setup(
    name='mlmodels',
    packages=find_packages(),
    version='0.1.1',
    description='Base model and transformer classes.',
    author='Jes Ravnbøl',
    license='',
    install_requires=[
        'click',
        'docker',
        'gorilla',
        'mlflow',
        'simplejson',
        'scikit-learn',
        'pandas',
        'jinja2',
        'PyYAML',
        'numpy',
        'pandera==0.3.0'
    ],
    package_data={'': [
        'model_service/Dockerfile',
        'model_service/requirements.txt',
        'model_service/app/routes/openapi/*',
        'openapi_templates/*',
    ]},
    entry_points='''
            [console_scripts]
            mlmodels=mlmodels.cli:cli
'''
)
