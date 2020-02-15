from pathlib import Path
import os
from distutils.dir_util import copy_tree
import tempfile
from subprocess import Popen, PIPE, STDOUT

filedir = os.path.dirname(os.path.realpath(__file__))


def copy_docker_dir_to_tmp_dir(tmp_dir_path):
    copy_tree(str(filedir), tmp_dir_path)


def build_model_service_docker_image(
        s3_model_uri,
        tag,
        model_version,
):

    with tempfile.TemporaryDirectory() as tmpdirname:

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
            f"s3_model_uri={s3_model_uri}",
            "--build-arg",
            f"MODEL_VERSION={model_version}",
        ],
            cwd=tmpdirname,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            print(x, end='')


