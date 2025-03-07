#!/bin/bash

# Define image and container names
IMAGE_NAME="pneumonia-classification"
CONTAINER_NAME="pneumonia-container"

# Build the Docker image
echo "Building Docker image..."
docker build . -t $IMAGE_NAME

# Check if a container with the same name is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the container
echo "Running the Docker container..."
docker run -it --rm -p 8888:8888 --name $CONTAINER_NAME $IMAGE_NAME
