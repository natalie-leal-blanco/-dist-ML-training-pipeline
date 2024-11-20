import boto3
import os
import sys

def setup_aws_resources():
    """Set up required AWS resources for testing"""
    try:
        s3 = boto3.client('s3')
        
        # Create buckets if they don't exist
        buckets = [
            os.getenv('TEST_DATA_BUCKET'),
            os.getenv('TEST_CHECKPOINT_BUCKET')
        ]
        
        for bucket in buckets:
            try:
                # For us-east-1, don't specify LocationConstraint
                s3.create_bucket(Bucket=bucket)
                print(f"Created bucket: {bucket}")
            except s3.exceptions.BucketAlreadyExists:
                print(f"Bucket already exists: {bucket}")
            except s3.exceptions.BucketAlreadyOwnedByYou:
                print(f"You already own bucket: {bucket}")
            
        # Upload dummy training data
        dummy_data = b"dummy data"
        s3.put_object(
            Bucket=os.getenv('TEST_DATA_BUCKET'),
            Key='train/class_0/dummy1.jpg',
            Body=dummy_data
        )
        s3.put_object(
            Bucket=os.getenv('TEST_DATA_BUCKET'),
            Key='val/class_0/dummy1.jpg',
            Body=dummy_data
        )
        
        print("AWS resources setup completed successfully")
        return True
    except Exception as e:
        print(f"Error setting up AWS resources: {e}")
        return False

if __name__ == "__main__":
    success = setup_aws_resources()
    sys.exit(0 if success else 1)