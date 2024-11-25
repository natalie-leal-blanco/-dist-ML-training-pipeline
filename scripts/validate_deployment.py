import boto3
import argparse
import yaml
from typing import Dict, List, Tuple
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
        
    def validate_s3_buckets(self) -> Tuple[bool, Dict[str, str]]:
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
                results[bucket] = "OK"
            except Exception as e:
                results[bucket] = str(e)
        
        return all(status == "OK" for status in results.values()), results

    def validate_ecs_cluster(self) -> Tuple[bool, str]:
        """Validate ECS cluster exists and is active"""
        try:
            response = self.clients['ecs'].describe_clusters(
                clusters=['ml-training-cluster']
            )
            if not response['clusters']:
                return False, "Cluster not found"
            cluster = response['clusters'][0]
            if cluster['status'] != 'ACTIVE':
                return False, f"Cluster status: {cluster['status']}"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def validate_cloudwatch_dashboard(self) -> Tuple[bool, str]:
        """Validate CloudWatch dashboard exists"""
        try:
            response = self.clients['cloudwatch'].get_dashboard(
                DashboardName='MLTrainingDashboard'
            )
            return True, "OK"
        except self.clients['cloudwatch'].exceptions.ResourceNotFound:
            return False, "Dashboard not found"
        except Exception as e:
            return False, str(e)

    def validate_cloudwatch_alarms(self) -> Tuple[bool, Dict[str, str]]:
        """Validate CloudWatch alarms exist"""
        results = {}
        for alert in self.config['monitoring']['alerts']:
            alarm_name = f"MLTraining_{alert['metric']}"
            try:
                response = self.clients['cloudwatch'].describe_alarms(
                    AlarmNames=[alarm_name]
                )
                if not response['MetricAlarms']:
                    results[alarm_name] = "Alarm not found"
                else:
                    results[alarm_name] = "OK"
            except Exception as e:
                results[alarm_name] = str(e)
        
        return all(status == "OK" for status in results.values()), results

    def run_all_validations(self) -> Dict[str, Dict]:
        """Run all validation checks with detailed results"""
        s3_success, s3_details = self.validate_s3_buckets()
        ecs_success, ecs_details = self.validate_ecs_cluster()
        dashboard_success, dashboard_details = self.validate_cloudwatch_dashboard()
        alarms_success, alarms_details = self.validate_cloudwatch_alarms()
        
        return {
            'S3 Buckets': {
                'success': s3_success,
                'details': s3_details
            },
            'ECS Cluster': {
                'success': ecs_success,
                'details': ecs_details
            },
            'CloudWatch Dashboard': {
                'success': dashboard_success,
                'details': dashboard_details
            },
            'CloudWatch Alarms': {
                'success': alarms_success,
                'details': alarms_details
            }
        }

def print_validation_results(results: Dict, verbose: bool = False):
    """Print validation results in a formatted way"""
    print("\nDeployment Validation Results:")
    print("=============================")
    all_success = True
    
    for component, result in results.items():
        status = '✅ PASS' if result['success'] else '❌ FAIL'
        print(f"{component}: {status}")
        
        if verbose or not result['success']:
            details = result['details']
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  - {key}: {value}")
            else:
                print(f"  - Details: {details}")
        
        if not result['success']:
            all_success = False
    
    print("\n" + ("✅ All validations passed successfully!" if all_success else "❌ Some validations failed. See details above."))
    return all_success

def main():
    parser = argparse.ArgumentParser(description='Validate ML infrastructure deployment')
    parser.add_argument('--config', required=True, help='Path to configuration YAML')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed results', default=False)
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1
    
    validator = DeploymentValidator(args.config)
    results = validator.run_all_validations()
    success = print_validation_results(results, args.verbose)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())