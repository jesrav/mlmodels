import os
from distutils.dir_util import copy_tree
import logging
import logging


def copy_model_from_s3(s3_path, out_dir):
    os.system(f'aws s3 cp --recursive {s3_path} {out_dir}')


def copy_model_from_local(in_dir, out_dir):
    logger = logging.getLogger(__name__)
    logger.info(f'copying model from local path {in_dir}')
    copy_tree(str(in_dir), out_dir)


def copy_model_from_model_uri(model_uri, out_dir):
    """Copy model folder from s3 path or local path.
    Parameters
    ----------
    model_uri : str
        local path or s3 path to model folder we are copying

    out_dir : str
        local path to folder we are copying model folder content into.
    """

    if model_uri.startswith('s3://'):
        copy_model_from_s3(s3_path=model_uri, out_dir=out_dir)
    else:
        copy_model_from_local(in_dir=model_uri, out_dir=out_dir)