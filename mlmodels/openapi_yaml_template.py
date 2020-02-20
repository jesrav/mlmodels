from jinja2 import Template
import yaml

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
    {% if feat in possible_categorical_column_values %}
                    enum: {{possible_categorical_column_values[feat]}}
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
    {% if target_col in possible_categorical_column_values %}
                        enum: {{possible_categorical_column_values[target_col]}}
    {% endif %}
{% endfor %}
            type: array
tags:
- predict
"""


def open_api_yaml_specification(
        model_input_record_field_schema_dict,
        possible_categorical_column_values,
        model_target_field_schema_dict):
    t = Template(template_str)
    return t.render(
        feature_dict=model_input_record_field_schema_dict,
        possible_categorical_column_values=possible_categorical_column_values,
        target_dict=model_target_field_schema_dict)


def open_api_dict_specification(
        model_input_record_field_schema_dict,
        possible_categorical_column_values,
        model_target_field_schema_dict,
):
    return yaml.safe_load(open_api_yaml_specification(
        model_input_record_field_schema_dict,
        possible_categorical_column_values,
        model_target_field_schema_dict,
    ))
