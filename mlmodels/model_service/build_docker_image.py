from pathlib import Path
import os
from distutils.dir_util import copy_tree
import tempfile
from subprocess import Popen, PIPE, STDOUT
from mlmodels.model_service.copy_model import copy_model_from_model_uri

file_dir = os.path.dirname(os.path.realpath(__file__))

model_path = Path('app/model')

def copy_docker_dir_to_tmp_dir(tmp_dir_path):
    copy_tree(str(file_dir), tmp_dir_path)


def build_model_service_docker_image(
        model_uri,
        model_version,
        tag,
):

    with tempfile.TemporaryDirectory() as tmpdirname:

        copy_model_from_model_uri(model_uri, str(Path(tmpdirname) / model_path))

        copy_docker_dir_to_tmp_dir(tmpdirname)

        proc = Popen([
            "docker",
            "build",
            "-t",
            tag,
            "-f",
            "Dockerfile",
            ".",
            "--build-arg",
            f"s3_model_uri={model_uri}",
            "--build-arg",
            f"MODEL_VERSION={model_version}",
        ],
            cwd=tmpdirname,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            print(x, end='')


