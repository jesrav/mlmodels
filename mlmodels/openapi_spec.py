from pathlib import Path
from typing import List

from jinja2 import Template
import yaml
from collections import namedtuple
from mlmodels.data_frame_schema import DataFrameSchema
import pkg_resources

_DTYPE_TO_JSON_TYPE_MAP = {
    'int64': {'type': 'number', 'format': 'integer'},
    'int32': {'type': 'number', 'format': 'integer'},
    'float64': {'type': 'number', 'format': 'float'},
    'float32': {'type': 'number', 'format': 'float'},
    'O': {'type': 'string'},
}

openapi_template_path = pkg_resources.resource_filename(
    'mlmodels',
    'openapi_templates/openapi_version2.yaml'
)

# Named tuple to render data frame column in jinja2
OpenAPICol = namedtuple("OpenAPICol", ["name", "format", "type", 'enum', 'min_', 'max_'])


def _data_frame_schema_to_open_api_cols(data_frame_schema: DataFrameSchema) -> List[OpenAPICol]:
    open_api_cols = []
    for _, col in data_frame_schema.column_dict.items():
        if hasattr(col.interval, 'start_value'):
            min_ = col.interval.start_value
        else:
            min_ = None
        if hasattr(col.interval, 'end_value'):
            max_ = col.interval.end_value
        else:
            max_ = None

        openapi_col = OpenAPICol(
            name=col.name,
            format=_DTYPE_TO_JSON_TYPE_MAP[col.dtype]['format'],
            type=_DTYPE_TO_JSON_TYPE_MAP[col.dtype]['type'],
            enum=col.enum,
            min_=min_,
            max_=max_,
        )
        open_api_cols.append(openapi_col)
    return open_api_cols


def open_api_yaml_specification_from_df_method(
    input_df_schema: DataFrameSchema,
    output_df_schema: DataFrameSchema,
) -> str:
    """Get open API spec for model from template in a YAML representation.

    Parameters
    ----------
    input_df_schema: DataFrameSchema
    output_df_schema: DataFrameSchema

    Returns
    -------
    str
        YAML representation of the open API spec for the the model predictions.
    """
    with open(openapi_template_path) as f:
        t = Template(f.read())
    return t.render(
        input_openapi_named_tuple=_data_frame_schema_to_open_api_cols(input_df_schema),
        output_openapi_named_tuple=_data_frame_schema_to_open_api_cols(output_df_schema),
    )


def open_api_dict_specification_from_df_method(
        input_df_schema: DataFrameSchema,
        output_df_schema: DataFrameSchema,
) -> str:
    """Get open API spec for model from template in a YAML representation.

    Parameters
    ----------
    input_df_schema: DataFrameSchema
    output_df_schema: DataFrameSchema

    Returns
    -------
    dict
        Dictionary representation of the open API spec for the the model predictions.
    """

    return yaml.safe_load(open_api_yaml_specification_from_df_method(
        input_df_schema,
        output_df_schema
    ))
