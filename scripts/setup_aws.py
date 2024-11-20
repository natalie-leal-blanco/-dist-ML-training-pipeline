import boto3
import os
import sys
from io import BytesIO
from PIL import Image
import numpy as np

def create_dummy_image():
    """Create a dummy RGB image"""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

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
                s3.create_bucket(Bucket=bucket)
                print(f"Created bucket: {bucket}")
            except s3.exceptions.BucketAlreadyExists:
                print(f"Bucket already exists: {bucket}")
            except s3.exceptions.BucketAlreadyOwnedByYou:
                print(f"You already own bucket: {bucket}")
        
        # Create dummy images
        dummy_image = create_dummy_image()
        
        # Upload multiple dummy files for training and validation
        for i in range(4):
            s3.put_object(
                Bucket=os.getenv('TEST_DATA_BUCKET'),
                Key=f'train/class_0/dummy{i}.jpg',
                Body=dummy_image,
                ContentType='image/jpeg'
            )
            s3.put_object(
                Bucket=os.getenv('TEST_DATA_BUCKET'),
                Key=f'val/class_0/dummy{i}.jpg',
                Body=dummy_image,
                ContentType='image/jpeg'
            )
        
        print("AWS resources setup completed successfully")
        return True
    except Exception as e:
        print(f"Error setting up AWS resources: {e}")
        return False

if __name__ == "__main__":
    success = setup_aws_resources()
    sys.exit(0 if success else 1)