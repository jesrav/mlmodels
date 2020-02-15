import os
import shutil
from subprocess import Popen
from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "app/model"

SUPPORTED_FLAVORS = [
    pyfunc.FLAVOR_NAME,
    mleap.FLAVOR_NAME
]


def _install_pyfunc_deps(model_path=None, install_mlflow=False):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    # If model is a pyfunc model, create its conda env (even if it also has mleap flavor)
    has_env = False
    if model_path:
        model_config_path = os.path.join(model_path, "MLmodel")
        model = Model.load(model_config_path)
        # NOTE: this differs from _serve cause we always activate the env even if you're serving
        # an mleap model
        if pyfunc.FLAVOR_NAME not in model.flavors:
            return
        conf = model.flavors[pyfunc.FLAVOR_NAME]
        if pyfunc.ENV in conf:
            print("creating and activating custom environment")
            env = conf[pyfunc.ENV]
            env_path_dst = os.path.join("/opt/mlflow/", env)
            env_path_dst_dir = os.path.dirname(env_path_dst)
            if not os.path.exists(env_path_dst_dir):
                os.makedirs(env_path_dst_dir)
            shutil.copyfile(os.path.join(MODEL_PATH, env), env_path_dst)
            conda_create_model_env = "conda env create -n custom_env -f {}".format(env_path_dst)
            if Popen(["bash", "-c", conda_create_model_env]).wait() != 0:
                raise Exception("Failed to create model environment.")
            has_env = True
    activate_cmd = ["source /miniconda/bin/activate custom_env"] if has_env else []
    # NB: install gunicorn[gevent] from pip rather than from conda because gunicorn is already
    # dependency of mlflow on pip and we expect mlflow to be part of the environment.
    install_server_deps = ["pip install gunicorn[gevent]"]
    if Popen(["bash", "-c", " && ".join(activate_cmd + install_server_deps)]).wait() != 0:
        raise Exception("Failed to install serving dependencies into the model environment.")
    if has_env and install_mlflow:
        install_mlflow_cmd = [
            "pip install /opt/mlflow/." if _container_includes_mlflow_source()
            else "pip install mlflow=={}".format(MLFLOW_VERSION)
        ]
        if Popen(["bash", "-c", " && ".join(activate_cmd + install_mlflow_cmd)]).wait() != 0:
            raise Exception("Failed to install mlflow into the model environment.")


def _container_includes_mlflow_source():
    return os.path.exists("/opt/mlflow/setup.py")


if __name__ == '__main__':
    _install_pyfunc_deps(MODEL_PATH)