# Configuration
$BackupDir = "C:\backups"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"
$DBName = $env:POSTGRES_DB ?? "cpas"
$DBUser = $env:POSTGRES_USER ?? "postgres"
$DBHost = "db"
$RetentionDays = 7

# Create backup directory
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

# Create database backup
Write-Host "Creating database backup..."
$env:PGPASSWORD = $env:POSTGRES_PASSWORD
pg_dump -h $DBHost -U $DBUser -d $DBName | Out-File -FilePath "$BackupDir\backup_$Date.sql" -Encoding UTF8
Compress-Archive -Path "$BackupDir\backup_$Date.sql" -DestinationPath "$BackupDir\backup_$Date.sql.zip"
Remove-Item "$BackupDir\backup_$Date.sql"

# Create Redis backup
Write-Host "Creating Redis backup..."
docker exec cpas_redis_1 redis-cli SAVE
docker cp cpas_redis_1:/data/dump.rdb "$BackupDir\redis_$Date.rdb"

# Create Prometheus backup
Write-Host "Creating Prometheus backup..."
docker exec cpas_prometheus_1 tar -czf /tmp/prometheus_backup.tar.gz /prometheus
docker cp cpas_prometheus_1:/tmp/prometheus_backup.tar.gz "$BackupDir\prometheus_$Date.tar.gz"
docker exec cpas_prometheus_1 rm /tmp/prometheus_backup.tar.gz

# Create Grafana backup
Write-Host "Creating Grafana backup..."
docker exec cpas_grafana_1 tar -czf /tmp/grafana_backup.tar.gz /var/lib/grafana
docker cp cpas_grafana_1:/tmp/grafana_backup.tar.gz "$BackupDir\grafana_$Date.tar.gz"
docker exec cpas_grafana_1 rm /tmp/grafana_backup.tar.gz

# Upload to S3 if configured
if ($env:S3_BUCKET -and $env:S3_REGION -and $env:S3_ACCESS_KEY -and $env:S3_SECRET_KEY) {
    Write-Host "Uploading backups to S3..."
    $env:AWS_ACCESS_KEY_ID = $env:S3_ACCESS_KEY
    $env:AWS_SECRET_ACCESS_KEY = $env:S3_SECRET_KEY
    $env:AWS_DEFAULT_REGION = $env:S3_REGION
    
    # Upload each backup file
    Get-ChildItem "$BackupDir\*_$Date.*" | ForEach-Object {
        aws s3 cp $_.FullName "s3://$env:S3_BUCKET/backups/$($_.Name)"
    }
    
    # Clean up old backups in S3
    $oldestDate = (Get-Date).AddDays(-$RetentionDays)
    aws s3 ls "s3://$env:S3_BUCKET/backups/" | ForEach-Object {
        $fileDate = [DateTime]::ParseExact($_.Substring(0, 16), "yyyy-MM-dd HH:mm", $null)
        if ($fileDate -lt $oldestDate) {
            $fileName = $_.Split(" ")[-1]
            aws s3 rm "s3://$env:S3_BUCKET/backups/$fileName"
        }
    }
}

# Clean up old backups locally
Write-Host "Cleaning up old backups..."
Get-ChildItem $BackupDir | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays) } | Remove-Item

Write-Host "Backup completed successfully!" -ForegroundColor Green 