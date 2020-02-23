from jinja2 import Template
import yaml
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
{% for feat in feature_dict %}
                {{feat}}:
                    format: {{feature_dict[feat]['format']}}
                    nullable: False
                    type: {{feature_dict[feat]['type']}}
    {% {feature_dict[feat]['enum'] %}
                    enum: {{feature_dict[feat]['enum']}}
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
          predictions:
            items:
                properties:
{% for target_col in target_dict %}
                    {{target_col}}:
                        format: {{target_dict[target_col]['format']}}
                        nullable: False
                        type: {{target_dict[target_col]['type']}}
    {% {target_dict[feat]['enum'] %}
                    enum: {{target_dict[feat]['enum']}}
    {% endif %}
{% endfor %}
            type: array
tags:
- predict
"""


def _data_frame_schema_to_dict(data_frame_schema):
    dict_ = {col.name: _DTYPE_TO_JSON_TYPE_MAP[col.dtype] for col in data_frame_schema.columns}
    for col in data_frame_schema.columns:
        dict_[col.name].update(enum=col.enum)
    return dict_


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
        feature_dict=_data_frame_schema_to_dict(feature_df_schema),
        target_dict=_data_frame_schema_to_dict(target_df_schema),
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
