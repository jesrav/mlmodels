import yaml
from pathlib import Path
import mlflow.pyfunc

MODEL_PATH = Path("app/model")
YAML_PATH = Path("app/routes/swagger/predict.yaml")


def load_model():
    return mlflow.pyfunc.load_model(str(MODEL_PATH))


def write_swagger_yaml_to_file(model, path_out):
    swagger_dict = model.python_model.model.get_open_api_dict()
    with open(str(path_out), 'w') as outfile:
        yaml.dump(swagger_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    write_swagger_yaml_to_file(load_model(), YAML_PATH)
