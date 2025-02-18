ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Create directories and set permissions
RUN mkdir -p ${YOLO_CONFIG_DIR} ${MPLCONFIGDIR} && \
    chmod 777 ${YOLO_CONFIG_DIR} ${MPLCONFIGDIR}

# Create non-root user with video group membership
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --ingroup video \
    --uid 10001 \
    appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Copy application files
COPY --chown=appuser:video . .

# Expose port if your application needs it
EXPOSE 8000

# Run the application
CMD ["python3", "main.py"]
