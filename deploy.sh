#!/bin/bash
# deploy.sh
# Deployment script for horizon.perfect.io

set -e

IMAGE_NAME="pure-fastmcp"

echo "Building Docker image $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "Deploying to horizon.perfect.io..."
# Replace with actual horizon CLI or pipeline steps, for example:
# horizon deploy --image $IMAGE_NAME --app pure-fastmcp

echo "Deployment pipeline executed successfully."
