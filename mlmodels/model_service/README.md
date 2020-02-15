# Model service
Build a dockerized web api with swagger validation from a custom MLFLOW python model stored in s3.
The custom model class that is wrapped as an mlflow.pyfunc model must have the following 
- get_open_api_dict: Method that returns an open api specification as a dictionary.
- model_input_from_dict: Method that transforms the dictionary model input, from the posted json, to input that can be passed to a predict method.
- MODEL_NAME: Attribute with a model name.
- model_initiated_dt: Attribute indicating when the object was initialized (when the model was trained).
The model must return predictions in an array-like form.   


Example usage
```console
# Build
docker build . --build-arg s3_model_uri="s3://<model-path>" \
--build-arg AWS_ACCESS_KEY_ID=<key> \
--build-arg AWS_SECRET_ACCESS_KEY=<key> \
--build-arg MODEL_VERSION=<key> \
-t model-service:latest

# Run
docker run -p 5000:5000 model-service:latest
```
