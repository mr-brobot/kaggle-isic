# ISIC 2024 - Skin Cancer Detection with 3D-TBP

https://www.kaggle.com/competitions/isic-2024-challenge

## Development

### 1. Configure Kaggle API key

https://www.kaggle.com/docs/api#getting-started-installation-&-authentication

### 2. Acquire data

```sh
mkdir data
cd data

kaggle competitions download -c isic-2024-challenge
unzip isic-2024-challenge.zip
rm isic-2024-challenge.zip

cd -
```

### 3. AWS Setup for OpenTelemetry/X-Ray Tracing

This project uses AWS X-Ray for distributed tracing via OTLP (OpenTelemetry Protocol).

#### 3.1 AWS Credentials

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

#### 3.2 One-Time AWS Account Setup

X-Ray's OTLP endpoint requires CloudWatch Logs as the trace destination. Run the setup script once per AWS account/region:

```bash
./scripts/setup-xray.sh
```

This script will:
1. Create CloudWatch Logs resource policy allowing X-Ray to write traces
2. Configure X-Ray to use CloudWatch Logs destination
3. Enable Transaction Search with 1% sampling (free tier)
4. Verify the configuration

**Note**: The destination may show `Status: PENDING` for up to 10 minutes before becoming `ACTIVE`.

#### 3.3 Where to Find Traces

After running `make bench`, traces will be available in:

1. **CloudWatch Logs**: `aws/spans` log group (created automatically, contains raw trace data)
2. **X-Ray Console**: CloudWatch → X-Ray traces → Traces
   - Filter by service name: `isic`
   - Look for spans named `load_batch`

**Note**: After the first `make bench` run, it may take 1-2 minutes for traces to appear in the console.
