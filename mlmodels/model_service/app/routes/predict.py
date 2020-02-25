import flask
from flask import request
import flasgger.utils
from app.ml_model import MLModel
import json

# Blueprint for the model_train_datetime endpoint
prediction_endpoint_bp = flask.Blueprint("predict_endpoint", __name__)

ml_model = MLModel()

import pandas as pd
@prediction_endpoint_bp.route("/predict", methods=["POST"])
@flasgger.utils.swag_from('swagger/predict.yaml', validation=True)
def predict():

    # Get posted input from api call    
    inputs = flask.request.get_json()

    # Make predictions
    response_json = ml_model.predict_from_dict(inputs)
    print(response_json)
    return response_json
