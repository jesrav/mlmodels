from pathlib import Path
from jinja2 import Template
import yaml
from collections import namedtuple
from mlmodels.data_frame_schema import DataFrameSchema

_DTYPE_TO_JSON_TYPE_MAP = {
    'int64': {'type': 'number', 'format': 'integer'},
    'int32': {'type': 'number', 'format': 'integer'},
    'float64': {'type': 'number', 'format': 'float'},
    'float32': {'type': 'number', 'format': 'float'},
    'O': {'type': 'string'},
}

openapi_template_path = Path('mlmodels/openapi_templates/openapi_version2.yaml')

# Named tuple to render data frame column in jinja
OpenAPICol = namedtuple("OpenAPICol", ["name", "format", "type", 'enum'])


def _data_frame_schema_to_open_api_cols(data_frame_schema):
    open_api_cols = [
        OpenAPICol(
            name=col.name,
            format=_DTYPE_TO_JSON_TYPE_MAP[col.dtype]['format'],
            type=_DTYPE_TO_JSON_TYPE_MAP[col.dtype]['type'],
            enum=col.enum,
        )
        for _, col in data_frame_schema.column_dict.items()
    ]
    return open_api_cols


def open_api_yaml_specification(
    feature_df_schema: DataFrameSchema,
    target_df_schema: DataFrameSchema,
) -> str:
    """Get open API spec for model from template in a YAML representation.

    Parameters
    ----------
    feature_df_schema: DataFrameSchema
    target_df_schema: DataFrameSchema

    Returns
    -------
    str
        YAML representation of the open API spec for the the model predictions.
    """
    with open('mlmodels/openapi_templates/openapi_version2.yaml') as f:
        t = Template(f.read())
    return t.render(
        feature_openapi_named_tuple=_data_frame_schema_to_open_api_cols(feature_df_schema),
        target_openapi_named_tuple=_data_frame_schema_to_open_api_cols(target_df_schema),
    )


def open_api_dict_specification(
        feature_df_schema: DataFrameSchema,
        target_df_schema: DataFrameSchema,
) -> str:
    """Get open API spec for model from template in a YAML representation.

    Parameters
    ----------
    feature_df_schema: DataFrameSchema
    target_df_schema: DataFrameSchema

    Returns
    -------
    dict
        Dictionary representation of the open API spec for the the model predictions.
    """

    return yaml.safe_load(open_api_yaml_specification(
        feature_df_schema,
        target_df_schema
    ))
