#!/usr/bin/env python3

import sys
import boto3
import argparse
import yaml
from pathlib import Path

def validate_resources(config_path):
    """Validate deployed AWS resources"""
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize AWS clients
        s3 = boto3.client('s3')
        ecs = boto3.client('ecs')
        cloudwatch = boto3.client('cloudwatch')

        print("\nDeployment Validation Results:")
        print("=============================")
        
        # Validate S3 buckets
        bucket_prefix = config['infrastructure']['storage']['s3_bucket_prefix']
        buckets = [
            f"{bucket_prefix}-data",
            f"{bucket_prefix}-checkpoints",
            f"{bucket_prefix}-logs"
        ]
        
        print("\nChecking S3 Buckets:")
        all_buckets_exist = True
        for bucket in buckets:
            try:
                s3.head_bucket(Bucket=bucket)
                print(f"✅ {bucket}: EXISTS")
            except Exception as e:
                print(f"❌ {bucket}: {str(e)}")
                all_buckets_exist = False

        # Validate ECS Cluster
        print("\nChecking ECS Cluster:")
        try:
            response = ecs.describe_clusters(clusters=['ml-training-cluster'])
            if response['clusters'] and response['clusters'][0]['status'] == 'ACTIVE':
                print("✅ ml-training-cluster: ACTIVE")
                cluster_exists = True
            else:
                print("❌ ml-training-cluster: NOT ACTIVE")
                cluster_exists = False
        except Exception as e:
            print(f"❌ ml-training-cluster: {str(e)}")
            cluster_exists = False

        # Validate CloudWatch Dashboard
        print("\nChecking CloudWatch Dashboard:")
        try:
            cloudwatch.get_dashboard(DashboardName='MLTrainingDashboard')
            print("✅ MLTrainingDashboard: EXISTS")
            dashboard_exists = True
        except Exception as e:
            print(f"❌ MLTrainingDashboard: {str(e)}")
            dashboard_exists = False

        # Validate CloudWatch Alarms
        print("\nChecking CloudWatch Alarms:")
        alarms_exist = True
        for alert in config['monitoring']['alerts']:
            alarm_name = f"MLTraining_{alert['metric']}"
            try:
                response = cloudwatch.describe_alarms(AlarmNames=[alarm_name])
                if response['MetricAlarms']:
                    print(f"✅ {alarm_name}: EXISTS")
                else:
                    print(f"❌ {alarm_name}: NOT FOUND")
                    alarms_exist = False
            except Exception as e:
                print(f"❌ {alarm_name}: {str(e)}")
                alarms_exist = False

        # Overall validation result
        print("\nValidation Summary:")
        print("==================")
        validations = {
            "S3 Buckets": all_buckets_exist,
            "ECS Cluster": cluster_exists,
            "CloudWatch Dashboard": dashboard_exists,
            "CloudWatch Alarms": alarms_exist
        }

        for resource, status in validations.items():
            print(f"{resource}: {'✅ PASS' if status else '❌ FAIL'}")

        all_passed = all(validations.values())
        print(f"\nOverall Status: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        
        return all_passed

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate ML infrastructure deployment')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed results')
    args = parser.parse_args()

    success = validate_resources(args.config)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())