# [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

## Dataset

This project uses the competition dataset via HuggingFace datasets. Review the Kaggle-HuggingFace conversion in the [Dataset notebook](nbs/dataset.ipynb). Review [the EDA notebook](nbs/eda.ipynb) to learn more about the dataset.

## OpenTelemetry

This project uses AWS X-Ray for distributed tracing via OTLP (OpenTelemetry Protocol).

### AWS Credentials

AWS credentials are automatically mounted from `~/.aws` into the devcontainer.

**Required IAM Permissions:**

- `xray:PutTraceSegments`
- `xray:PutTelemetryRecords`
- `xray:UpdateTraceSegmentDestination`
- `xray:GetTraceSegmentDestination`
- `xray:UpdateIndexingRule`
- `logs:PutResourcePolicy`
- `logs:CreateLogGroup`

Or use managed policy: `AWSXrayFullAccess` (for setup) + `CloudWatchLogsFullAccess`

**Note:** The AWS region is auto-detected from your AWS configuration. If no region is configured, traces will not be sent to X-Ray.

### One-Time AWS Account Setup

X-Ray's OTLP endpoint requires CloudWatch Logs as the trace destination. Run the setup script once per AWS account/region:

```bash
./scripts/setup-xray.sh
```

This script will:

1. Create CloudWatch Logs resource policy allowing X-Ray to write traces
2. Configure X-Ray to use CloudWatch Logs destination
3. Enable Transaction Search with 1% sampling (free tier)

**Note**: The destination may show `Status: PENDING` for up to 10 minutes before becoming `ACTIVE`.

### Where to Find Traces

After running `make bench`, traces will be available in:

1. **CloudWatch Logs**: `aws/spans` log group (created automatically, contains raw trace data)
2. **X-Ray Console**: CloudWatch → X-Ray traces → Traces
   - Filter by service name: `isic`
   - Look for spans named `load_batch`

**Note**: After the first `make bench` run, it may take 1-2 minutes for traces to appear in the console.
