FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Set environment variable for port
ENV PORT=5000

# Run the application
CMD gunicorn --chdir /app/backend --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 main:app
