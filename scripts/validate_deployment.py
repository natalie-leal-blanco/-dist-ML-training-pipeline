import boto3
import argparse
import yaml
from typing import Dict, List
import json
from pathlib import Path

class DeploymentValidator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.clients = {
            's3': boto3.client('s3'),
            'ecs': boto3.client('ecs'),
            'cloudwatch': boto3.client('cloudwatch'),
            'logs': boto3.client('logs')
        }
        
    def validate_s3_buckets(self) -> Dict[str, bool]:
        """Validate S3 buckets exist and are accessible"""
        bucket_prefix = self.config['infrastructure']['storage']['s3_bucket_prefix']
        expected_buckets = [
            f"{bucket_prefix}-data",
            f"{bucket_prefix}-checkpoints",
            f"{bucket_prefix}-logs"
        ]
        
        results = {}
        for bucket in expected_buckets:
            try:
                self.clients['s3'].head_bucket(Bucket=bucket)
                results[bucket] = True
            except Exception as e:
                print(f"Error checking bucket {bucket}: {e}")
                results[bucket] = False
        return results

    def validate_ecs_cluster(self) -> bool:
        """Validate ECS cluster exists and is active"""
        try:
            response = self.clients['ecs'].describe_clusters(
                clusters=['ml-training-cluster']
            )
            cluster = response['clusters'][0]
            return cluster['status'] == 'ACTIVE'
        except Exception as e:
            print(f"Error checking ECS cluster: {e}")
            return False

    def validate_cloudwatch_dashboard(self) -> bool:
        """Validate CloudWatch dashboard exists"""
        try:
            response = self.clients['cloudwatch'].get_dashboard(
                DashboardName='MLTrainingDashboard'
            )
            return True
        except Exception as e:
            print(f"Error checking dashboard: {e}")
            return False

    def validate_cloudwatch_alarms(self) -> Dict[str, bool]:
        """Validate CloudWatch alarms exist"""
        results = {}
        for alert in self.config['monitoring']['alerts']:
            alarm_name = f"MLTraining_{alert['metric']}"
            try:
                response = self.clients['cloudwatch'].describe_alarms(
                    AlarmNames=[alarm_name]
                )
                results[alarm_name] = len(response['MetricAlarms']) > 0
            except Exception as e:
                print(f"Error checking alarm {alarm_name}: {e}")
                results[alarm_name] = False
        return results

    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation checks"""
        results = {
            'S3 Buckets': all(self.validate_s3_buckets().values()),
            'ECS Cluster': self.validate_ecs_cluster(),
            'CloudWatch Dashboard': self.validate_cloudwatch_dashboard(),
            'CloudWatch Alarms': all(self.validate_cloudwatch_alarms().values())
        }
        return results

def main():
    parser = argparse.ArgumentParser(description='Validate ML infrastructure deployment')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    args = parser.parse_args()
    
    validator = DeploymentValidator(args.config)
    results = validator.run_all_validations()
    
    print("\nDeployment Validation Results:")
    print("=============================")
    for component, status in results.items():
        status_str = '✅ PASS' if status else '❌ FAIL'
        print(f"{component}: {status_str}")
    
    if not all(results.values()):
        print("\n⚠️ Some validations failed. Please check the logs above for details.")
        return 1
    else:
        print("\n✅ All validations passed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())