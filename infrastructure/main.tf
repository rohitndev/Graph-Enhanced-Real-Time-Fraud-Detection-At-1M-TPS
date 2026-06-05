# ─────────────────────────────────────────────────────────────
# Graph-Enhanced Real-Time Fraud Detection — AWS infrastructure
#
# Provisions the cloud resources the pipeline materialises to:
#   * S3 data lake (scores / rings / narratives / model artifacts)
#   * Kinesis stream for high-risk fraud alerts
#   * MSK (Kafka) serverless cluster for the 1M TPS transaction stream
#   * IAM user + policy for programmatic access from the pipeline
# ─────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  # Credentials come from the standard AWS provider chain
  # (env vars, shared credentials file, or an instance/role profile).
}

# ── S3 data lake ──────────────────────────────────────────────
resource "aws_s3_bucket" "fraud_lake" {
  bucket = var.s3_bucket_name
  tags = {
    Project = "graph-enhanced-fraud-detection"
    Env     = var.app_environment
  }
}

resource "aws_s3_bucket_versioning" "fraud_lake" {
  bucket = aws_s3_bucket.fraud_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "fraud_lake" {
  bucket = aws_s3_bucket.fraud_lake.id
  rule {
    id     = "expire-raw-scores"
    status = "Enabled"
    filter {
      prefix = "fraud-detection/scores/"
    }
    expiration {
      days = 90
    }
  }
}

# ── Kinesis stream for high-risk fraud alerts ─────────────────
resource "aws_kinesis_stream" "fraud_alerts" {
  name             = var.kinesis_stream_name
  shard_count      = var.kinesis_shard_count
  retention_period = 24
  stream_mode_details {
    stream_mode = "PROVISIONED"
  }
  tags = {
    Project = "graph-enhanced-fraud-detection"
    Env     = var.app_environment
  }
}

# ── MSK Serverless (Kafka) for the transaction stream ─────────
resource "aws_msk_serverless_cluster" "transactions" {
  count        = var.enable_msk ? 1 : 0
  cluster_name = "${var.app_name}-transactions"

  vpc_config {
    subnet_ids         = var.msk_subnet_ids
    security_group_ids = var.msk_security_group_ids
  }

  client_authentication {
    sasl {
      iam {
        enabled = true
      }
    }
  }
}

# ── IAM user + least-privilege policy for the pipeline ────────
resource "aws_iam_user" "pipeline" {
  name = "${var.app_name}-pipeline"
}

resource "aws_iam_user_policy" "pipeline" {
  name = "${var.app_name}-pipeline-policy"
  user = aws_iam_user.pipeline.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "S3DataLakeAccess"
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.fraud_lake.arn,
          "${aws_s3_bucket.fraud_lake.arn}/*"
        ]
      },
      {
        Sid      = "KinesisAlertAccess"
        Effect   = "Allow"
        Action   = ["kinesis:PutRecord", "kinesis:PutRecords", "kinesis:DescribeStream"]
        Resource = [aws_kinesis_stream.fraud_alerts.arn]
      }
    ]
  })
}
