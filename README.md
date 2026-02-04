## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- .github
|   `-- workflows
|       `-- ci.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- churn
|   |   |-- data
|   |   |   `-- raw
|   |   |       `-- churn.csv
|   |   |-- scripts
|   |   |   |-- evaluate.py
|   |   |   `-- preprocess.py
|   |   |-- __init__.py
|   |   `-- pipeline.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving the customer churn prediction problem. The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/churn/pipeline.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CI/CD system via GitHub Actions.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

A description of some of the artifacts is provided below:
<br/><br/>
Your GitHub Actions CI/CD workflow. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CI/CD system. The workflow is triggered on pushes to main/master branches and pull requests. You will need to configure the following GitHub Secrets:
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
- `AWS_DEFAULT_REGION`: Your AWS region (e.g., `us-east-1`)
- `SAGEMAKER_ROLE_ARN`: The IAM role ARN for SageMaker Pipeline execution

You can customize the workflow file as required, including other pipeline parameters.

```
|-- .github
|   `-- workflows
|       `-- ci.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the accuracy of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- churn
|   |   |-- data
|   |   |   `-- raw
|   |   |       `-- churn.csv
|   |   |-- scripts
|   |   |   |-- evaluate.py
|   |   |   `-- preprocess.py
|   |   |-- __init__.py
|   |   `-- pipeline.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Dataset for the Customer Churn Prediction Pipeline

This project uses a customer churn dataset to predict whether customers will churn. The dataset is located at `pipelines/churn/data/raw/churn.csv`. The pipeline includes preprocessing, model training with hyperparameter tuning, evaluation, and conditional model registration based on accuracy thresholds.

The pipeline will automatically upload the data to S3 and execute the full ML workflow including:
- Data preprocessing and train/validation/test split
- Model training with XGBoost and hyperparameter tuning
- Model evaluation
- Conditional model registration (only if accuracy meets the threshold)
