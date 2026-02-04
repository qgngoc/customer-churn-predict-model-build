
import os
import json
import logging
from typing import Optional

import boto3
import sagemaker
from sagemaker.session import Session

# Workflow / pipeline related imports
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet


# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----- Constants -----
PREFIX = "demo_churn_predict"
PIPELINE_NAME = "demo-churn-pred-pipeline"
MODEL_GROUP = "demo-churn-registry"

# Local paths (relative to repository root)
CUSTOMER_CHURN_DATA_LOCAL_PATH = "pipelines/churn/data/raw/churn.csv"
PREPROCESS_SCRIPT_LOCAL_PATH = "pipelines/churn/scripts/preprocess.py"
EVALUATE_SCRIPT_LOCAL_PATH = "pipelines/churn/scripts/evaluate.py"


# ----- Helpers -----

def get_session_role_bucket(session: Optional[Session] = None):
    session = session or sagemaker.session.Session()

    try:
        role = sagemaker.get_execution_role()
    except ValueError as e:
        # Local execution
        role = os.environ.get(
            "SAGEMAKER_ROLE_ARN",
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        )

    bucket = session.default_bucket()
    return session, role, bucket


def upload_raw_data(session: Session, local_path: str, prefix: str = PREFIX) -> str:
    """Upload local data file to S3 and return its URI."""
    uri = session.upload_data(path=local_path, key_prefix=f"{prefix}/data")
    logger.info("Customer Churn data uploaded to: %s", uri)
    return uri


# ----- Pipeline builder -----

def build_pipeline(
    session: Session,
    role: str,
    bucket: str,
    prefix: str = PREFIX,
    pipeline_name: str = PIPELINE_NAME,
) -> Pipeline:
    """Build and return a SageMaker Pipeline object."""

    region = session.boto_region_name

    # Pipeline parameters
    model_registry_package = ParameterString(
        name="ModelGroup", default_value="customer-churn-predict-registry"
    )

    input_data = ParameterString(
        name="InputData", default_value=f"s3://{bucket}/{prefix}/data/churn.csv"
    )

    preprocess_script = ParameterString(
        name="PreprocessScript", default_value=f"scripts/preprocess.py"
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    evaluate_script = ParameterString(
        name="EvaluateScript", default_value=f"scripts/evaluate.py"
    )

    max_training_jobs = ParameterInteger(name="MaxiumTrainingJobs", default_value=1)
    max_parallel_training_jobs = ParameterInteger(
        name="MaxiumParallelTrainingJobs", default_value=1
    )

    accuracy_condition_threshold = ParameterFloat(
        name="AccuracyConditionThreshold", default_value=0.7
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.t3.medium"
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.t2.medium"
    )

    # Processing step (preprocessing)
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
    )

    step_preprocess_data = ProcessingStep(
        name="Preprocess-Data",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket}", prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, "train"],
                ),
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket}", prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, "validation"],
                ),
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket}", prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, "test"],
                ),
            ),
        ],
        code=PREPROCESS_SCRIPT_LOCAL_PATH,
    )

    # Training and tuning
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-2",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    xgb_estimator = Estimator(
        image_uri=image_uri,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role,
        disable_profiler=True,
    )

    xgb_tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges={
            "eta": ContinuousParameter(0, 0.5),
            "alpha": ContinuousParameter(0, 1000),
            "min_child_weight": ContinuousParameter(1, 120),
            "max_depth": IntegerParameter(1, 10),
            "num_round": IntegerParameter(1, 2000),
            "subsample": ContinuousParameter(0.5, 1),
        },
        max_jobs=max_training_jobs,
        max_parallel_jobs=max_parallel_training_jobs,
    )

    step_tuning = TuningStep(
        name="Train-And-Tune-Model",
        tuner=xgb_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Evaluation
    evaluate_model_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
    )

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    step_evaluate_model = ProcessingStep(
        name="Evaluate-Model",
        processor=evaluate_model_processor,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=bucket),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket}", prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, "evaluation-report"],
                ),
            )
        ],
        code=EVALUATE_SCRIPT_LOCAL_PATH,
        property_files=[evaluation_report],
    )

    # Model registration
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    step_register_model = RegisterModel(
        name="Register-Model",
        estimator=xgb_estimator,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=bucket),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_registry_package,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate_model.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",
        ),
        right=accuracy_condition_threshold,
    )

    step_cond = ConditionStep(
        name="Accuracy-Condition",
        conditions=[cond_gte],
        if_steps=[step_register_model],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            input_data,
            preprocess_script,
            evaluate_script,
            model_approval_status,
            accuracy_condition_threshold,
            model_registry_package,
            max_parallel_training_jobs,
            max_training_jobs,
        ],
        steps=[step_preprocess_data, step_tuning, step_evaluate_model, step_cond],
    )

    return pipeline

def get_pipeline():
    session, role, bucket = get_session_role_bucket()

    # Upload raw CSV to S3 (if needed)
    upload_raw_data(session, CUSTOMER_CHURN_DATA_LOCAL_PATH, PREFIX)

    pipeline = build_pipeline(session=session, role=role, bucket=bucket, prefix=PREFIX, pipeline_name=PIPELINE_NAME)
    return pipeline

    # Validate pipeline definition and upsert the pipeline
    # json.loads(pipeline.definition())
    # logger.info("Pipeline definition created successfully")

    # # Submit (create or update) the pipeline in SageMaker
    # pipeline.upsert(role_arn=role)
    # logger.info("Pipeline upserted: %s", pipeline.name)

if __name__ == "__main__":
    session, role, bucket = get_session_role_bucket()
    logger.info("Successfully authorized with AWS")
    logger.info("Session: %s", session)
    logger.info("Role: %s", role)
    logger.info("Bucket: %s", bucket)
    pipeline = get_pipeline()
    logger.info("Pipeline definition created successfully")
    pipeline_definition = json.loads(pipeline.definition())
    logger.info("Pipeline definition:\n%s", json.dumps(pipeline_definition, indent=2))
