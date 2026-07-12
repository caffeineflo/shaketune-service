FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/caffeineflo/shaketune-service"
LABEL org.opencontainers.image.description="Shake&Tune Analysis Service for Klipper 3D printers"
LABEL org.opencontainers.image.licenses="GPL-3.0"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        fastapi==0.135.2 \
        jinja2==3.1.6 \
        python-multipart==0.0.22 \
        uvicorn==0.42.0

# Copy Shake&Tune module
COPY shaketune /app/shaketune/

# Copy minimal Klipper dependencies for shaper calibration
COPY service/klipper /app/service/klipper/

# Copy server and templates
COPY service/server.py /app/service/
COPY service/templates /app/service/templates/

# Run analysis without root privileges. The results bind mount must be writable
# by uid/gid 1000 on the host.
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid 1000 --create-home app && \
    mkdir -p /app/results /tmp/matplotlib && \
    chown -R app:app /app/results /tmp/matplotlib

# Environment variables
ENV RESULTS_DIR=/app/results
ENV KLIPPER_DIR=/app/service/klipper
ENV PYTHONPATH=/app
ENV HOME=/home/app

EXPOSE 8080

VOLUME /app/results

USER app

CMD ["uvicorn", "service.server:app", "--host", "0.0.0.0", "--port", "8080"]
