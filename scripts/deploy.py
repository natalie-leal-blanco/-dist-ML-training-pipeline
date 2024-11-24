import boto3
import argparse
import yaml
import sys
import os
from pathlib import Path
import json
import time

def create_service_linked_role():
    """Create the ECS service-linked role if it doesn't exist"""
    try:
        iam = boto3.client('iam')
        try:
            iam.get_role(RoleName='AWSServiceRoleForECS')
            print("ECS service-linked role already exists")
            return True
        except iam.exceptions.NoSuchEntityException:
            iam.create_service_linked_role(
                AWSServiceName='ecs.amazonaws.com'
            )
            print("Created ECS service-linked role")
            time.sleep(10)
            return True
    except Exception as e:
        print(f"Error creating service-linked role: {e}")
        return False

def create_dashboard_body(config, region):
    """Create properly formatted dashboard JSON"""
    widgets = []
    
    # Training metrics widget
    training_metrics = [metric['name'] for metric in config['monitoring']['metrics']]
    widgets.append({
        "type": "metric",
        "x": 0,
        "y": 0,
        "width": 12,
        "height": 6,
        "properties": {
            "metrics": [
                ["MLTraining", metric] for metric in training_metrics
            ],
            "period": 300,
            "stat": "Average",
            "region": region,
            "title": "Training Metrics"
        }
    })
    
    # GPU Utilization widget
    widgets.append({
        "type": "metric",
        "x": 0,
        "y": 6,
        "width": 12,
        "height": 6,
        "properties": {
            "metrics": [
                ["MLTraining", "gpu_utilization"]
            ],
            "period": 60,
            "stat": "Average",
            "region": region,
            "title": "GPU Utilization"
        }
    })
    
    dashboard = {
        "widgets": widgets
    }
    
    return json.dumps(dashboard)

def create_cloudwatch_alarms(cloudwatch_client, config):
    """Create CloudWatch alarms with proper numeric thresholds"""
    try:
        for alert in config['monitoring']['alerts']:
            # Parse threshold value properly
            condition_parts = alert['condition'].split()
            operator = condition_parts[0]
            # Remove any '%' sign and convert to float
            threshold = float(condition_parts[1].replace('%', ''))
            
            cloudwatch_client.put_metric_alarm(
                AlarmName=f"MLTraining_{alert['metric']}",
                MetricName=alert['metric'],
                Namespace='MLTraining',
                Period=alert.get('window', 300),
                EvaluationPeriods=2,
                Threshold=threshold,
                ComparisonOperator='GreaterThanThreshold' if operator == '>' else 'LessThanThreshold',
                Statistic='Average',
                ActionsEnabled=True,
                Dimensions=[{"Name": "Environment", "Value": "Production"}]
            )
            print(f"Created alarm for {alert['metric']}")
        return True
    except Exception as e:
        print(f"Error creating alarms: {e}")
        print(f"Alert details: {alert}")  # Print the problematic alert
        return False

def deploy_training_infrastructure(config_path: str):
    """Deploy the ML training infrastructure"""
    try:
        # Load and validate configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize AWS clients
        s3 = boto3.client('s3')
        cloudwatch = boto3.client('cloudwatch')
        ecs = boto3.client('ecs')
        
        # Create service-linked role first
        if not create_service_linked_role():
            raise Exception("Failed to create service-linked role")

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
            except Exception as e:
                print(f"Error creating bucket {bucket}: {e}")
                raise e

        # Create ECS cluster with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cluster_response = ecs.create_cluster(
                    clusterName='ml-training-cluster',
                    capacityProviders=['FARGATE_SPOT'],
                    defaultCapacityProviderStrategy=[{
                        'capacityProvider': 'FARGATE_SPOT',
                        'weight': 1
                    }]
                )
                print("Created ECS cluster")
                break
            except ecs.exceptions.InvalidParameterException as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Retrying ECS cluster creation... (attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
        
        # Create CloudWatch dashboard
        dashboard_body = create_dashboard_body(config, config['infrastructure']['region'])
        try:
            cloudwatch.put_dashboard(
                DashboardName='MLTrainingDashboard',
                DashboardBody=dashboard_body
            )
            print("Created CloudWatch dashboard")
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            print(f"Dashboard body: {dashboard_body}")
            raise e

        # Create CloudWatch alarms
        if not create_cloudwatch_alarms(cloudwatch, config):
            raise Exception("Failed to create CloudWatch alarms")
        
        print("\nDeployment completed successfully!")
        print("\nResources created:")
        print("- ECS Cluster: ml-training-cluster")
        print("- CloudWatch Dashboard: MLTrainingDashboard")
        print(f"- S3 Buckets: {', '.join(buckets)}")
        print("- CloudWatch Alarms for metrics")
        return True
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Deploy ML infrastructure')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    success = deploy_training_infrastructure(args.config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()