"""
Policyholder Lapse Prediction — Batch Inference Script
======================================================
Scores policyholders for lapse risk and assigns risk tiers.

Usage:
    python predict.py \
        --model-name insurance-lapse-prediction \
        --input-path s3://bucket/data/new-customers.csv \
        --output-path s3://bucket/predictions/
        --threshold 0.30
"""

import argparse
import boto3
import sagemaker
import pandas as pd
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Score policyholders for lapse risk')
    parser.add_argument('--model-package-group', type=str, default='insurance-lapse-prediction',
                        help='Model Registry package group name')
    parser.add_argument('--input-path', type=str, required=True,
                        help='S3 path to input CSV (no header, target in first column)')
    parser.add_argument('--output-path', type=str, required=True,
                        help='S3 path for output predictions')
    parser.add_argument('--threshold', type=float, default=0.30,
                        help='Classification threshold (default: 0.30)')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                        help='Instance type for Batch Transform')
    return parser.parse_args()


def get_approved_model(sm_client, model_package_group):
    """Retrieve the latest approved model from the Model Registry."""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response['ModelPackageSummaryList']:
        raise ValueError(f'No approved models found in {model_package_group}')
    
    package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
    details = sm_client.describe_model_package(ModelPackageName=package_arn)
    
    model_data_url = details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    image_uri = details['InferenceSpecification']['Containers'][0]['Image']
    version = details['ModelPackageVersion']
    
    print(f'Using model: {model_package_group} v{version}')
    print(f'Model artifact: {model_data_url}')
    
    return image_uri, model_data_url, version


def run_batch_transform(session, sm_client, image_uri, model_data_url, 
                        input_path, output_path, instance_type, role):
    """Run Batch Transform and return the output S3 path."""
    from sagemaker.model import Model
    from sagemaker.transformer import Transformer
    
    model_name = f'lapse-scoring-{int(time.time())}'
    
    model = Model(
        image_uri=image_uri,
        model_data=model_data_url,
        role=role,
        sagemaker_session=session,
        name=model_name
    )
    model.create(instance_type=instance_type)
    
    transformer = Transformer(
        model_name=model_name,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        sagemaker_session=session,
        assemble_with='Line',
        accept='text/csv'
    )
    
    print(f'Starting Batch Transform job...')
    transformer.transform(
        data=input_path,
        content_type='text/csv',
        split_type='Line',
        input_filter='$[1:]',
        join_source='Input'
    )
    transformer.wait()
    print('Batch Transform complete.')
    
    # Clean up model
    sm_client.delete_model(ModelName=model_name)
    
    return transformer.output_path


def assign_risk_tiers(predictions_prob, threshold):
    """Assign risk tiers based on prediction probabilities."""
    tiers = pd.cut(
        predictions_prob,
        bins=[0.0, threshold, 0.60, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    return tiers


def main():
    args = parse_args()
    
    # Setup
    session = sagemaker.Session()
    sm_client = boto3.client('sagemaker')
    role = sagemaker.get_execution_role()
    
    print('═══ Policyholder Lapse Prediction ═══')
    print(f'Input: {args.input_path}')
    print(f'Output: {args.output_path}')
    print(f'Threshold: {args.threshold}')
    print()
    
    # Step 1: Get the latest approved model from the Registry
    image_uri, model_data_url, version = get_approved_model(
        sm_client, args.model_package_group
    )
    
    # Step 2: Run Batch Transform
    output_path = run_batch_transform(
        session, sm_client, image_uri, model_data_url,
        args.input_path, args.output_path, args.instance_type, role
    )
    
    print(f'\nScoring complete. Results saved to: {output_path}')
    print(f'Model version: {args.model_package_group} v{version}')
    print(f'Threshold: {args.threshold}')
    print('═══ Done ═══')


if __name__ == '__main__':
    main()