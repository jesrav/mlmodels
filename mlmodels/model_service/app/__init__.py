from pathlib import Path
import os
import logging
from flask import Flask
from flasgger import Swagger
# Import blueprints
from app.routes.predict import prediction_endpoint_bp
from app.routes.model_train_datetime import model_train_datetime_endpoint_bp
from app.routes.model_version import model_version_endpoint_bp
from app.ml_model import MLModel
# Import config
from config import get_config

swagger = Swagger()


def instantiate_ml_model():
    """ This function runs at application startup and loads the ml model"""
    logging.info("Load ml model")
    ml_model = MLModel()
    ml_model.load_model()
    ml_model.set_model_version(os.getenv("MODEL_VERSION"))


def create_app(enviroment, model_version):
    logging.info("Init app")
    app = Flask(__name__)

    logging.info("Load configuration")
    app.config.from_object(get_config(enviroment, model_version))
    logging.info(f"App running in {enviroment} with model version {model_version}")

    instantiate_ml_model()

    logging.info("Register prediction endpoint")
    app.register_blueprint(prediction_endpoint_bp)

    logging.info("Register model train datetime endpoint")
    app.register_blueprint(model_train_datetime_endpoint_bp)

    logging.info("Register model version endpoint")
    app.register_blueprint(model_version_endpoint_bp)

    logging.info("Initialize swagger")
    swagger.init_app(app)

    logging.info("Running app")

    return app



