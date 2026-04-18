FROM python:3.10-slim

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /app

# Copy requirements first
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY --chown=user . .

# Download SyncNet model
RUN mkdir -p syncnet_python/data && \
    wget -q http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model \
    -O syncnet_python/data/syncnet_v2.model || echo "Model download failed - will retry at runtime"

# Expose port
EXPOSE 7860

# Run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]