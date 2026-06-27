# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    DATABASE_PATH=/app/data/sylvi_profile.db

# Set working directory
WORKDIR /app

# Install uv globally
RUN pip install --no-cache-dir uv

# Copy project files
COPY . /app

# Sync project dependencies (creates .venv and installs packages)
RUN uv sync --frozen

# Create data directory and set permissions for Hugging Face non-root user (1000)
RUN mkdir -p /app/data /app/temp_downloads && \
    chmod -R 777 /app

# Expose port
EXPOSE 7860

# Run main.py using uv's environment
CMD ["/app/.venv/bin/python", "main.py"]
