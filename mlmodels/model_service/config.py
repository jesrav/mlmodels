import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    SWAGGER = {
    'title': 'ML model api',
    'swagger': '2.0',
    }

class Development(Config):
    ENVIRONMENT = 'development'
    DEBUG = True


class Staging(Config):
    ENVIROMENT = 'staging'
    DEBUG = False


class Production(Config):
    ENVIROMENT = 'production'
    DEBUG = False


def get_config(enviroment, model_version):

    if enviroment == "production":
        Environment = Production
    elif enviroment == "staging":
        Environment = Staging
    else:
        Environment = Development

    Environment.MODEL_VERSION = model_version

    return Environment
