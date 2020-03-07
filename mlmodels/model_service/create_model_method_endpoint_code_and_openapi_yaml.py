import yaml
from jinja2 import Template
from pathlib import Path
import logging
import mlflow.pyfunc

MODEL_PATH = Path("app/model")
OPENAPI_YAML_DIR = Path("app/routes/openapi/")
MODEL_METHOD_ENDPOINT_CODE_PATH = Path("app/routes/model_method_endpoints.py")


model_method_enpoint_template = """
import flask
import flasgger.utils
from app.ml_model import MLModel

# Blueprint for the model_train_datetime endpoint
model_method_endpoints_bp = flask.Blueprint("model_method_endpoints", __name__)

ml_model = MLModel()

{% for method_name in method_names %}
@model_method_endpoints_bp.route("/{{ method_name }}", methods=["POST"])
@flasgger.utils.swag_from('openapi/{{ method_name }}.yaml', validation=True)
def {{method_name}}():

    # Get posted input from api call    
    inputs = flask.request.get_json()
    # Make predictions
    response_json = ml_model.method_call_from_dict_input("{{ method_name }}",  inputs)
    return response_json

{% endfor %}
"""


def load_model():
    return mlflow.pyfunc.load_model(str(MODEL_PATH))


def write_openapi_yaml_to_file(model: mlflow.pyfunc):
    model_method_names = list(model.python_model.model.model_method_schema_dict.keys())
    for model_method_name in model_method_names:
        open_dict = model.python_model.model.get_method_open_api_dict(model_method_name)
        with open(OPENAPI_YAML_DIR / Path(f'{model_method_name}.yaml'), 'w') as outfile:
            yaml.dump(open_dict, outfile, default_flow_style=False)


def create_model_method_endpoint_code(model: mlflow.pyfunc, path_out: str):
    model_method_names = list(model.python_model.model.model_method_schema_dict.keys())
    t = Template(model_method_enpoint_template)
    outfile_text = t.render(method_names = model_method_names)
    with open(Path(path_out), "w") as f:
        f.write(outfile_text)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logging.info("Loading model.")
    model = load_model()
    logging.info("Writing open api yaml files.")
    write_openapi_yaml_to_file(model)
    logging.info("Writing model method endpoint code.")
    create_model_method_endpoint_code(model, MODEL_METHOD_ENDPOINT_CODE_PATH)
