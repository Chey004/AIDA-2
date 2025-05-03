#!/bin/bash

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME=${POSTGRES_DB:-cpas}
DB_USER=${POSTGRES_USER:-postgres}
DB_HOST="db"
RETENTION_DAYS=7

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create database backup
echo "Creating database backup..."
PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Create Redis backup
echo "Creating Redis backup..."
docker exec cpas_redis_1 redis-cli SAVE
docker cp cpas_redis_1:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Create Prometheus backup
echo "Creating Prometheus backup..."
docker exec cpas_prometheus_1 tar -czf /tmp/prometheus_backup.tar.gz /prometheus
docker cp cpas_prometheus_1:/tmp/prometheus_backup.tar.gz $BACKUP_DIR/prometheus_$DATE.tar.gz
docker exec cpas_prometheus_1 rm /tmp/prometheus_backup.tar.gz

# Create Grafana backup
echo "Creating Grafana backup..."
docker exec cpas_grafana_1 tar -czf /tmp/grafana_backup.tar.gz /var/lib/grafana
docker cp cpas_grafana_1:/tmp/grafana_backup.tar.gz $BACKUP_DIR/grafana_$DATE.tar.gz
docker exec cpas_grafana_1 rm /tmp/grafana_backup.tar.gz

# Upload to S3 if configured
if [ ! -z "$S3_BUCKET" ] && [ ! -z "$S3_REGION" ] && [ ! -z "$S3_ACCESS_KEY" ] && [ ! -z "$S3_SECRET_KEY" ]; then
    echo "Uploading backups to S3..."
    export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
    export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY
    export AWS_DEFAULT_REGION=$S3_REGION
    
    # Upload each backup file
    for file in $BACKUP_DIR/*_$DATE.*; do
        aws s3 cp $file s3://$S3_BUCKET/backups/$(basename $file)
    done
    
    # Clean up old backups in S3
    aws s3 ls s3://$S3_BUCKET/backups/ | while read -r line; do
        createDate=$(echo $line | awk {'print $1" "$2'})
        createDate=$(date -d "$createDate" +%s)
        olderThan=$(date -d "-$RETENTION_DAYS days" +%s)
        if [[ $createDate -lt $olderThan ]]; then
            fileName=$(echo $line | awk {'print $4'})
            aws s3 rm s3://$S3_BUCKET/backups/$fileName
        fi
    done
fi

# Clean up old backups locally
echo "Cleaning up old backups..."
find $BACKUP_DIR -type f -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully!" 