#!/bin/bash

# Configuration
ENV_FILE=".env.production"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="/backups"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(cat $ENV_FILE | grep -v '^#' | xargs)
else
    echo "Error: $ENV_FILE not found"
    exit 1
fi

# Create backup directory
mkdir -p $BACKUP_DIR

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down

# Pull latest images
echo "Pulling latest images..."
docker-compose -f $COMPOSE_FILE pull

# Build and start containers
echo "Starting containers..."
docker-compose -f $COMPOSE_FILE up -d --build

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check service health
echo "Checking service health..."
curl -f http://localhost/health || {
    echo "Health check failed"
    exit 1
}

echo "Deployment completed successfully!" 