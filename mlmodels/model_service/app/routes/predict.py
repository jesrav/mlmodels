import flask
import flasgger.utils
from app.ml_model import MLModel

# Blueprint for the model_train_datetime endpoint
prediction_endpoint_bp = flask.Blueprint("predict_endpoint", __name__)

ml_model = MLModel()


@prediction_endpoint_bp.route("/predict", methods=["POST"])
@flasgger.utils.swag_from('swagger/predict.yaml', validation=True)
def predict():

    # Get posted input from api call    
    inputs = flask.request.get_json()
    # Make predictions
    response_json = ml_model.predict_from_dict(inputs)
    return response_json
