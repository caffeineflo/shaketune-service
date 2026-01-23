FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/caffeineflo/shaketune-service"
LABEL org.opencontainers.image.description="Shake&Tune Analysis Service for Klipper 3D printers"
LABEL org.opencontainers.image.licenses="GPL-3.0"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart jinja2

# Copy Shake&Tune module
COPY shaketune /app/shaketune/

# Copy minimal Klipper dependencies for shaper calibration
COPY service/klipper /app/service/klipper/

# Copy server and templates
COPY service/server.py /app/service/
COPY service/templates /app/service/templates/

# Create results directory
RUN mkdir -p /app/results

# Environment variables
ENV RESULTS_DIR=/app/results
ENV KLIPPER_DIR=/app/service/klipper
ENV PYTHONPATH=/app

EXPOSE 8080

VOLUME /app/results

CMD ["uvicorn", "service.server:app", "--host", "0.0.0.0", "--port", "8080"]
