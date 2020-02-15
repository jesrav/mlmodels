import flask
import flasgger.utils
from app.ml_model import MLModel

# Blueprint for the model_train_datetime endpoint
model_train_datetime_endpoint_bp = flask.Blueprint("model_train_datetime_endpoint", __name__)

ml_model = MLModel()


@model_train_datetime_endpoint_bp.route("/model_train_datetime", methods=["GET"])
@flasgger.utils.swag_from('swagger/model_train_datetime.yaml')
def model_train_datetime():
    response = {'model_train_datetime': str(ml_model.get_model_train_datetime())}
    response_json = flask.jsonify(response)
    return response_json
