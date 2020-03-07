FROM python:3.7
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         build-essential \
         cmake \
         git-core \
    && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:$PATH"

# Set up the program in the image
WORKDIR /opt/mlflow

# Install MLFLOW
RUN pip install mlflow

# copy folder into docker container
COPY app /opt/mlflow/app
COPY api.py /opt/mlflow/api.py
COPY install_mlfow_dependencies.py /opt/mlflow/install_mlfow_dependencies.py
COPY create_model_method_endpoint_code_and_openapi_yaml.py /opt/mlflow/create_predict_endpoint_openapi_yaml.py
COPY config.py /opt/mlflow/config.py
COPY requirements.txt /opt/mlflow/requirements.txt

# Install model specific dependencies
#RUN pip install awscli

ARG s3_model_uri
ARG MODEL_VERSION
ENV MODEL_VERSION=$MODEL_VERSION

RUN python install_mlfow_dependencies.py

RUN echo "source activate custom_env" > ~/.bashrc
ENV PATH /miniconda/envs/custom_env/bin:$PATH

RUN conda install pip

# Install model serving dependencies
RUN /miniconda/envs/custom_env/bin/pip install -r requirements.txt

# Create yaml file for prediction endpoint openapi specification
RUN python create_predict_endpoint_openapi_yaml.py

ENV FLASK_APP=api.py

EXPOSE 5000

ENTRYPOINT [ "gunicorn", "--threads", "4", "-b", ":5000", "--access-logfile", "-", "--error-logfile", "-" , "api:app" ]