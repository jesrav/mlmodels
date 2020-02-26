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

template_str = """
requestBody:
  required: true
  content:
    application/json:
      schema:
        properties:
          data:
            items:
              properties:
{% for feat in feature_openapi_named_tuple %}
                {{ feat.name }}:
                    format: {{ feat.format }}
                    nullable: False
                    type: {{ feat.type }}
    {% if feat.enum %}
                    enum: {{ feat.enum }}
    {% endif %}
{% endfor %}
            type: array
        required:
          - data
        type: object

responses:
  200:
    description: List of predictions
    content:
      application/json:
        schema:
          items:
            properties:
{% for target in target_openapi_named_tuple %}
              {{ target.name }}:
                format: {{ target.format }}
                nullable: False
                type: {{ target.type }}
    {% if target.enum %}
                enum: {{ target.enum }}
    {% endif %}
{% endfor %}
          type: array
tags:
- predict
"""

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

    t = Template(template_str)
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
