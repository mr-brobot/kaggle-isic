#!/bin/bash
set -euo pipefail

# Setup AWS X-Ray with CloudWatch Logs for OTLP tracing
# This script configures the one-time AWS account/region setup required for X-Ray OTLP endpoint

echo "=== AWS X-Ray OTLP Setup ==="
echo

# Get AWS account ID and region
echo "→ Detecting AWS account and region..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

if [ -z "$REGION" ]; then
    echo "❌ Error: AWS region not configured"
    echo "   Run: aws configure set region <your-region>"
    exit 1
fi

echo "✓ Account ID: $ACCOUNT_ID"
echo "✓ Region: $REGION"
echo

# 1. Create CloudWatch Logs resource policy for X-Ray
echo "→ Creating CloudWatch Logs resource policy..."
aws logs put-resource-policy \
  --policy-name XRayCloudWatchLogsPolicy \
  --policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "XRayWriteAccess",
    "Effect": "Allow",
    "Principal": { "Service": "xray.amazonaws.com" },
    "Action": ["logs:PutLogEvents", "logs:CreateLogStream"],
    "Resource": [
      "arn:aws:logs:'"$REGION"':'"$ACCOUNT_ID"':log-group:aws/spans:*",
      "arn:aws:logs:'"$REGION"':'"$ACCOUNT_ID"':log-group:/aws/application-signals/data:*"
    ],
    "Condition": {
      "ArnLike": { "aws:SourceArn": "arn:aws:xray:'"$REGION"':'"$ACCOUNT_ID"':*" },
      "StringEquals": { "aws:SourceAccount": "'"$ACCOUNT_ID"'" }
    }
  }]
}' --region "$REGION" > /dev/null

echo "✓ Resource policy created"
echo

# 2. Update X-Ray to use CloudWatch Logs destination
echo "→ Configuring X-Ray trace segment destination to CloudWatch Logs..."
RESULT=$(aws xray update-trace-segment-destination --destination "CloudWatchLogs" --region "$REGION")
STATUS=$(echo "$RESULT" | jq -r '.Status')

echo "✓ Update initiated (Status: $STATUS)"

if [ "$STATUS" = "PENDING" ]; then
    echo "  ⏳ Transition to ACTIVE may take up to 10 minutes"
fi
echo

# 3. Enable Transaction Search (optional, 1% sampling is free tier)
echo "→ Enabling Transaction Search with 1% sampling..."
aws xray update-indexing-rule \
  --name "Default" \
  --rule '{"Probabilistic": {"DesiredSamplingPercentage": 1.0}}' \
  --region "$REGION" > /dev/null

echo "✓ Transaction Search enabled"
echo

# 4. Verify configuration
echo "→ Verifying configuration..."
DESTINATION=$(aws xray get-trace-segment-destination --region "$REGION")
DEST_TYPE=$(echo "$DESTINATION" | jq -r '.Destination')
DEST_STATUS=$(echo "$DESTINATION" | jq -r '.Status')

echo "✓ Destination: $DEST_TYPE"
echo "✓ Status: $DEST_STATUS"
echo

if [ "$DEST_STATUS" = "ACTIVE" ]; then
    echo "✅ Setup complete! X-Ray is ready to receive traces."
    echo
    echo "Next steps:"
    echo "  1. Run: make bench"
    echo "  2. View traces in CloudWatch Logs: aws/spans log group"
    echo "  3. View traces in X-Ray Console: CloudWatch → X-Ray traces → Traces"
elif [ "$DEST_STATUS" = "PENDING" ]; then
    echo "⏳ Setup initiated successfully."
    echo
    echo "The destination is still activating. This can take up to 10 minutes."
    echo "Check status with: aws xray get-trace-segment-destination --region $REGION"
else
    echo "⚠️  Unexpected status: $DEST_STATUS"
    echo "Please check AWS Console or run: aws xray get-trace-segment-destination --region $REGION"
    exit 1
fi
