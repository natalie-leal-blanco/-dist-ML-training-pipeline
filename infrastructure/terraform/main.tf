terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket = "tf-state-ml-pipeline"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

# S3 bucket for training data
resource "aws_s3_bucket" "training_data" {
  bucket = "ml-training-data-${random_string.bucket_suffix.result}"
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  lower   = true
}

# ECS Cluster for training
resource "aws_ecs_cluster" "training_cluster" {
  name = "ml-training-cluster"
}

# SQS Queue for job management
resource "aws_sqs_queue" "training_queue" {
  name = "training-jobs-queue"
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "training_logs" {
  name              = "/ecs/ml-training"
  retention_in_days = 14
}