import yaml
import boto3
import argparse
from typing import Dict, Any
import sys

class DeploymentVerifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aws_clients = {
            's3': boto3.client('s3'),
            'cloudwatch': boto3.client('cloudwatch'),
            'logs': boto3.client('logs')
        }
    
    def verify_s3_buckets(self) -> bool:
        """Verify S3 buckets exist and are accessible"""
        try:
            prefix = self.config['infrastructure']['storage']['s3_bucket_prefix']
            buckets = [
                f"{prefix}-data",
                f"{prefix}-checkpoints",
                f"{prefix}-logs"
            ]
            
            response = self.aws_clients['s3'].list_buckets()
            existing_buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            for bucket in buckets:
                if bucket not in existing_buckets:
                    print(f"Missing bucket: {bucket}")
                    return False
            return True
        except Exception as e:
            print(f"Error verifying S3 buckets: {e}")
            return False
    
    def verify_cloudwatch_metrics(self) -> bool:
        """Verify CloudWatch metrics are properly configured"""
        try:
            for metric in self.config['monitoring']['metrics']:
                response = self.aws_clients['cloudwatch'].list_metrics(
                    MetricName=metric['name'],
                    Namespace='MLTraining'
                )
                if not response['Metrics']:
                    print(f"Missing metric: {metric['name']}")
                    return False
            return True
        except Exception as e:
            print(f"Error verifying CloudWatch metrics: {e}")
            return False
    
    def verify_logging(self) -> bool:
        """Verify logging configuration"""
        try:
            log_group = self.config['logging']['cloudwatch']['log_group']
            response = self.aws_clients['logs'].describe_log_groups(
                logGroupNamePrefix=log_group
            )
            if not response['logGroups']:
                print(f"Missing log group: {log_group}")
                return False
            return True
        except Exception as e:
            print(f"Error verifying logging: {e}")
            return False
    
    def verify_all(self) -> Dict[str, bool]:
        """Run all verifications"""
        results = {
            'S3 Buckets': self.verify_s3_buckets(),
            'CloudWatch Metrics': self.verify_cloudwatch_metrics(),
            'Logging': self.verify_logging()
        }
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to production configuration YAML')
    args = parser.parse_args()
    
    verifier = DeploymentVerifier(args.config)
    results = verifier.verify_all()
    
    print("\nDeployment Verification Results:")
    print("================================")
    for component, status in results.items():
        print(f"{component}: {'✅ PASS' if status else '❌ FAIL'}")
    
    # Exit with success only if all verifications pass
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()