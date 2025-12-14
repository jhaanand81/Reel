FROM python:3.11-slim

# Install FFmpeg, espeak-ng (required for Kokoro TTS), and audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch CPU-only version first (smaller image, no GPU needed)
# This is required by Kokoro TTS for neural audio synthesis
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Kokoro TTS models during build (not runtime)
# This downloads the 82M model and voices (~200MB total)
# IMPORTANT: This MUST succeed for TTS to work - no silent failures!
RUN python -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); print('[KOKORO] Model loaded successfully!')"

# Copy the rest of the application
COPY . .

# Create required directories for outputs and data
RUN mkdir -p /app/backend/outputs/videos /app/backend/outputs/audio /app/backend/outputs/scripts /app/backend/data

# Set permissions
RUN chmod -R 777 /app/backend/outputs /app/backend/data

# Expose port
EXPOSE 5000

# Set environment variable for port
ENV PORT=5000

# Run the application
CMD gunicorn --chdir /app/backend --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 main:app
