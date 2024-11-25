#!/usr/bin/env python3

import subprocess
import sys
import time

def run_step(command, description):
    """Run a command and handle its result"""
    print(f"\n🚀 {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} failed")
        return False
    print(f"✅ {description} completed")
    return True

def main():
    print("\n📦 ML Training Infrastructure Setup")
    print("=================================")
    
    steps = [
        ("python scripts/verify_credentials.py", "Verifying AWS credentials"),
        ("aws configure list", "Checking AWS configuration"),
        ("python scripts/create_iam_policy.py", "Setting up IAM policy"),
        ("sleep 10", "Waiting for policy to propagate"),
        ("python scripts/deploy.py --config config/production.yml", "Deploying infrastructure"),
        ("sleep 5", "Waiting for resources to propagate"),
        ("python scripts/validate_deployment.py --config config/production.yml", "Validating deployment")
    ]
    
    for command, description in steps:
        if not run_step(command, description):
            print(f"\n❌ Setup failed at: {description}")
            print("\nTroubleshooting steps:")
            print("1. Check the error messages above")
            print("2. Verify your AWS credentials")
            print("3. Check resource permissions")
            print("4. Try running the failed step individually")
            sys.exit(1)
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()