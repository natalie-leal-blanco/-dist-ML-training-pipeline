import boto3
import argparse
import yaml
from typing import List
import sys

def cleanup_resources(config_path: str, force: bool = False):
    """Clean up AWS resources created by deployment"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize AWS clients
        s3 = boto3.client('s3')
        ecs = boto3.client('ecs')
        cloudwatch = boto3.client('cloudwatch')
        
        if not force:
            confirm = input("⚠️ This will delete all resources. Are you sure? (y/N): ")
            if confirm.lower() != 'y':
                print("Cleanup cancelled.")
                return False

        # Clean up S3 buckets
        bucket_prefix = config['infrastructure']['storage']['s3_bucket_prefix']
        buckets = [
            f"{bucket_prefix}-data",
            f"{bucket_prefix}-checkpoints",
            f"{bucket_prefix}-logs"
        ]
        
        for bucket in buckets:
            try:
                # Delete all objects first
                response = s3.list_objects_v2(Bucket=bucket)
                if 'Contents' in response:
                    objects = [{'Key': obj['Key']} for obj in response['Contents']]
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': objects}
                    )
                # Delete bucket
                s3.delete_bucket(Bucket=bucket)
                print(f"Deleted bucket: {bucket}")
            except Exception as e:
                print(f"Error deleting bucket {bucket}: {e}")

        # Delete ECS cluster
        try:
            ecs.delete_cluster(cluster='ml-training-cluster')
            print("Deleted ECS cluster")
        except Exception as e:
            print(f"Error deleting ECS cluster: {e}")

        # Delete CloudWatch dashboard
        try:
            cloudwatch.delete_dashboards(DashboardNames=['MLTrainingDashboard'])
            print("Deleted CloudWatch dashboard")
        except Exception as e:
            print(f"Error deleting dashboard: {e}")

        # Delete CloudWatch alarms
        try:
            for alert in config['monitoring']['alerts']:
                alarm_name = f"MLTraining_{alert['metric']}"
                cloudwatch.delete_alarms(AlarmNames=[alarm_name])
                print(f"Deleted alarm: {alarm_name}")
        except Exception as e:
            print(f"Error deleting alarms: {e}")

        print("\nCleanup completed successfully!")
        return True

    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean up ML infrastructure resources')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    success = cleanup_resources(args.config, args.force)
    sys.exit(0 if success else 1)