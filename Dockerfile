FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Download SyncNet model
RUN python -c "import urllib.request; urllib.request.urlretrieve('http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model', 'syncnet_python/data/syncnet_v2.model')"

# Expose port
EXPOSE 7860

# Run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]