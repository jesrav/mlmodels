from jinja2 import Template
import yaml

template_str = """
tags:
  - ml_endpoint
info:
  title: Prediction open api spec
  version: 1.0.0
openapi: 3.0.2
parameters:
- in: "body"
  name: "data"
  description: "List of feature records."
  required: true
  schema:
    $ref: "#/definitions/prediction_input"
responses:
  200:
    description: "List of predictions"
    name: predictions
    schema:
      $ref: "#/definitions/predictions"

definitions:
  prediction_input:
      properties:
        data:
          items:
            properties:
{% for feat in feature_dict %}
                {{feat}}:
                    format: {{feature_dict[feat]['format']}}
                    nullable: False
                    type: {{feature_dict[feat]['type']}}
{% endfor %}
          type: array
      required:
      - data
      type: object
  
  predictions:
      properties:
        predictions:
          items:
            format: {{target_dict['format']}}
            nullable: False
            type: {{target_dict['type']}}
          type: array
        type: object  
"""


def open_api_yaml_specification(feature_dict, target_dict):
    t = Template(template_str)
    return t.render(feature_dict=feature_dict, target_dict=target_dict)


def open_api_dict_specification(feature_dict, target_dict):
    return yaml.load(open_api_yaml_specification(feature_dict, target_dict))
