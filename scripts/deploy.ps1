# Configuration
$EnvFile = ".env.production"
$ComposeFile = "docker-compose.prod.yml"
$BackupDir = "C:\backups"

# Load environment variables
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^#].+?)=(.+)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], [System.EnvironmentVariableTarget]::Process)
        }
    }
} else {
    Write-Host "Error: $EnvFile not found" -ForegroundColor Red
    exit 1
}

# Create backup directory
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

# Stop existing containers
Write-Host "Stopping existing containers..."
docker-compose -f $ComposeFile down

# Pull latest images
Write-Host "Pulling latest images..."
docker-compose -f $ComposeFile pull

# Build and start containers
Write-Host "Starting containers..."
docker-compose -f $ComposeFile up -d --build

# Wait for services to be ready
Write-Host "Waiting for services to be ready..."
Start-Sleep -Seconds 30

# Check service health
Write-Host "Checking service health..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "Health check passed" -ForegroundColor Green
    } else {
        throw "Health check failed with status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "Health check failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "Deployment completed successfully!" -ForegroundColor Green 