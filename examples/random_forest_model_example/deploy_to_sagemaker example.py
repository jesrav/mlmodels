import pandas as pd
import mlflow.sagemaker as mfs
from dotenv import load_dotenv, find_dotenv
import json
import boto3


load_dotenv(find_dotenv())

model_uri="s3://smart-data-mlflow-artifacts/artifacts/3/8c27ecca16e046718348e1b217e234e6/artifacts/model/"

region = "eu-central-1"
image_ecr_url = "109317894517.dkr.ecr.eu-central-1.amazonaws.com/mlflow-pyfunc"
execution_role_arn = 'arn:aws:iam::109317894517:role/service-role/AmazonSageMaker-ExecutionRole-20190727T165273'

app_name = "mlflow-model-test6"
mfs.deploy(
    app_name=app_name,
    model_uri=model_uri,
    image_url=image_ecr_url,
    region_name=region,
    execution_role_arn=execution_role_arn,
    mode="replace",
    archive=True
)

# Query sagemaker endpoint

# Get data sample
csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(csv_url, sep=';')

# The predicted column is "quality" which is a scalar from [3, 9]
X = data.drop(["quality", "pH"], axis=1)
y = data["quality"]

# Test input validation
X = X.drop("fixed acidity", axis=1)

# Convert the sample input dataframe into a JSON-serialized pandas dataframe using the `split` orientation
input_json = X.head().to_json(orient="split")


def query_endpoint(app_name, input_json):
    client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType='application/json; format=pandas-split',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds


print("Sending batch prediction request with input dataframe json: {}".format(input_json))

# Evaluate the input by posting it to the deployed model
prediction = query_endpoint(app_name=app_name, input_json=input_json)

print(prediction)
