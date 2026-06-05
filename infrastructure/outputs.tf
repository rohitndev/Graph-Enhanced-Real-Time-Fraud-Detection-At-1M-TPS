# Outputs surfaced after `terraform apply` — wire these into your `.env`.

output "s3_bucket" {
  value       = aws_s3_bucket.fraud_lake.bucket
  description = "Set AWS_S3_BUCKET to this value."
}

output "kinesis_stream" {
  value       = aws_kinesis_stream.fraud_alerts.name
  description = "Set AWS_KINESIS_STREAM to this value."
}

output "pipeline_iam_user" {
  value       = aws_iam_user.pipeline.name
  description = "IAM user the pipeline authenticates as."
}

output "msk_bootstrap_endpoint" {
  value       = try(aws_msk_serverless_cluster.transactions[0].bootstrap_brokers_sasl_iam, "msk-disabled")
  description = "Set KAFKA_BOOTSTRAP_SERVERS to this value when MSK is enabled."
}
