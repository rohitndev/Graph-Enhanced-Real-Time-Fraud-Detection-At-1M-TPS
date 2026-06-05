# Input variables for the AWS infrastructure.

variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region to deploy into."
}

variable "app_name" {
  type        = string
  default     = "fraud-detection"
  description = "Application name prefix (no spaces)."
}

variable "app_environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment: dev / staging / prod."
}

variable "s3_bucket_name" {
  type        = string
  description = "Globally unique S3 bucket name for the fraud data lake."
}

variable "kinesis_stream_name" {
  type        = string
  default     = "fraud-alerts"
  description = "Kinesis stream name for high-risk fraud alerts."
}

variable "kinesis_shard_count" {
  type        = number
  default     = 1
  description = "Number of Kinesis shards (scale with alert volume)."
}

variable "enable_msk" {
  type        = bool
  default     = false
  description = "Whether to provision an MSK Serverless (Kafka) cluster."
}

variable "msk_subnet_ids" {
  type        = list(string)
  default     = []
  description = "Subnet IDs for the MSK cluster (required when enable_msk=true)."
}

variable "msk_security_group_ids" {
  type        = list(string)
  default     = []
  description = "Security group IDs for the MSK cluster."
}
