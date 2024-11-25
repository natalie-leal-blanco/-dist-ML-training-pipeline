import boto3
import argparse
import yaml
import sys
import os
from pathlib import Path

def validate_config(config):
    """Validate the configuration file"""
    required_sections = ['infrastructure', 'training', 'monitoring', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    return True

def deploy_training_infrastructure(config_path: str):
    """Deploy the training infrastructure using AWS"""
    try:
        # Load and validate configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)

        # Initialize AWS clients
        ecs = boto3.client('ecs')
        cloudwatch = boto3.client('cloudwatch')
        s3 = boto3.client('s3')
        
        # Create S3 buckets
        bucket_prefix = config['infrastructure']['storage']['s3_bucket_prefix']
        buckets = [
            f"{bucket_prefix}-data",
            f"{bucket_prefix}-checkpoints",
            f"{bucket_prefix}-logs"
        ]
        
        for bucket in buckets:
            try:
                s3.create_bucket(Bucket=bucket)
                print(f"Created bucket: {bucket}")
            except s3.exceptions.BucketAlreadyExists:
                print(f"Bucket already exists: {bucket}")
        
        # Create ECS cluster
        cluster_response = ecs.create_cluster(
            clusterName='ml-training-cluster',
            capacityProviders=['FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[{
                'capacityProvider': 'FARGATE_SPOT',
                'weight': 1
            }]
        )
        
        # Set up CloudWatch dashboard
        dashboard_body = {
            'widgets': [
                {
                    'type': 'metric',
                    'properties': {
                        'metrics': [
                            ['MLTraining', metric['name']]
                            for metric in config['monitoring']['metrics']
                        ],
                        'period': 300,
                        'stat': 'Average',
                        'region': config['infrastructure']['region'],
                        'title': 'Training Metrics'
                    }
                }
            ]
        }
        
        cloudwatch.put_dashboard(
            DashboardName='MLTrainingDashboard',
            DashboardBody=str(dashboard_body)
        )
        
        # Set up CloudWatch alerts
        for alert in config['monitoring']['alerts']:
            cloudwatch.put_metric_alarm(
                AlarmName=f"MLTraining_{alert['metric']}",
                MetricName=alert['metric'],
                Namespace='MLTraining',
                Period=alert.get('window', 300),
                EvaluationPeriods=2,
                Threshold=float(alert['condition'].split()[1]),
                ComparisonOperator='GreaterThanThreshold' if '>' in alert['condition'] else 'LessThanThreshold',
                ActionsEnabled=True
            )
        
        print("Deployment completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy ML training infrastructure')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    success = deploy_training_infrastructure(str(config_path))
    sys.exit(0 if success else 1)