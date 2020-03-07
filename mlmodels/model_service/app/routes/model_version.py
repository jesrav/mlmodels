import flask
import flasgger.utils
from app.ml_model import MLModel

# Blueprint for the model_version endpoint
model_version_endpoint_bp = flask.Blueprint("model_version_endpoint", __name__)

ml_model = MLModel()


@model_version_endpoint_bp.route("/model_version", methods=["GET"])
@flasgger.utils.swag_from('openapi/model_version.yaml')
def model_version():
    response = {'model_version': ml_model.get_model_version()}
    response_json = flask.jsonify(response)
    return response_json
