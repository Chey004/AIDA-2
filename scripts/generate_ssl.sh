#!/bin/bash

# Configuration
DOMAIN="yourdomain.com"
EMAIL="admin@yourdomain.com"
SSL_DIR="nginx/ssl"

# Create SSL directory if it doesn't exist
mkdir -p $SSL_DIR

# Generate self-signed certificate (for development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout $SSL_DIR/key.pem \
  -out $SSL_DIR/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"

# Set proper permissions
chmod 600 $SSL_DIR/key.pem
chmod 644 $SSL_DIR/cert.pem

echo "SSL certificates generated in $SSL_DIR" 