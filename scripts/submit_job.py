from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
import os 
from dotenv import load_dotenv
from azure.ai.ml.entities import Environment

load_dotenv()

def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv('SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('RESOURCE_GROUP'),
        workspace_name=os.getenv('WORKSPACE_NAME')
    )

    custom_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file='conda.yaml',
        name='acmg-variant-env'
    )

    job = command(
        code="./Hybrid-MLOps-Variant-Classifier/src",
        command="python train_model.py --data_folder ${{inputs.training_data}} --best_run_id ${{inputs.run_id}}",
        inputs = {
            "training_data" : Input(
                type=AssetTypes.URI_FOLDER,
                path="./Hybrid-MLOps-Variant-Classifier/src/data"
            ),
            "run_id" : "86c1fafe-0478-420b-b01c-8df4d6fdcb6e"
        },
        environment = custom_env,
        compute = os.getenv("COMPUTE_NAME"),
        display_name = 'acmg-xgb-training-run',
        experiment_name = 'acmg-variant-classification' 
    )
    returned_job = ml_client.create_or_update(job)

if __name__ == '__main__':
    main()