# Use an official lightweight Python runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for OpenCV and PIL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire WardrobeGenie project into the container
COPY . .

# Expose the port FastAPI attribute_predictor on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]