# Shake&Tune Analysis Service

A lightweight Docker-based web service that processes Klipper accelerometer data and generates input shaper calibration graphs. Designed for Creality K1 and other resource-constrained printers where running graph generation locally isn't practical.

## Why?

The K1's MIPS-based CPU struggles with the heavy matplotlib/numpy processing required for Shake&Tune graphs. This service offloads the analysis to a more capable machine on your local network, while keeping the simple accelerometer data collection on the printer.

## Quick Start

### 1. Run the Service

On any machine on your local network (NAS, Raspberry Pi, desktop):

```bash
# Using Docker Compose (recommended)
curl -O https://raw.githubusercontent.com/caffeineflo/shaketune-service/main/docker-compose.yml
docker compose up -d

# Or directly with Docker
docker run -d \
  --name shaketune-service \
  -p 8080:8080 \
  -v shaketune-results:/app/results \
  ghcr.io/caffeineflo/shaketune-service:latest
```

### 2. Configure Your K1

1. Copy `macros/shaketune.cfg` to your printer's config directory
2. Edit the `_SHAKETUNE_CONFIG` macro to set your Docker host's IP:
   ```ini
   [gcode_macro _SHAKETUNE_CONFIG]
   variable_host: "192.168.1.100"  # <-- Your Docker host IP
   variable_port: "8080"
   ```
3. Add to your `printer.cfg`:
   ```ini
   [include shaketune.cfg]
   ```
4. Restart Klipper

### 3. Run Calibration

From Mainsail/Fluidd console:

```gcode
SHAKETUNE_SHAPER    # Input shaper calibration
SHAKETUNE_BELTS     # Belt tension comparison
```

The console will display a URL to your graph when complete.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shaper` | POST | Upload resonance CSVs, get shaper graph |
| `/belts` | POST | Upload belt test CSVs, get comparison graph |
| `/vibrations` | POST | Upload vibration CSVs, get analysis graph |
| `/results/{file}` | GET | Retrieve generated graph |
| `/health` | GET | Health check |

### Example API Usage

```bash
# Analyze shaper resonances
curl -X POST http://localhost:8080/shaper \
  -F "files=@raw_data_x_x.csv" \
  -F "files=@raw_data_y_y.csv"

# Response:
# {"url": "/results/20241204_123456_abc123_shaper.png", "id": "abc123", "type": "shaper"}
```

## Architecture

```
┌─────────────────┐     CSV files      ┌─────────────────────┐
│                 │ ─────────────────▶ │                     │
│   Creality K1   │                    │  Shake&Tune Service │
│   (MIPS CPU)    │ ◀───────────────── │   (x86/ARM host)    │
│                 │    Graph URL       │                     │
└─────────────────┘                    └─────────────────────┘
        │                                       │
        │ TEST_RESONANCES                       │ numpy/matplotlib
        │ (lightweight)                         │ (heavy processing)
        ▼                                       ▼
   /tmp/*.csv                           /app/results/*.png
```

## Building Locally

```bash
# Build the image
docker build -t shaketune-service -f service/Dockerfile .

# Run locally
docker run -p 8080:8080 -v $(pwd)/results:/app/results shaketune-service
```

## Persistent Storage

Results are stored in a Docker volume by default. To use a local directory:

```yaml
# docker-compose.yml
services:
  shaketune:
    volumes:
      - ./results:/app/results  # Local directory instead of volume
```

## Network Security

The service is designed for local network use only. It:
- Binds to all interfaces (0.0.0.0) within the container
- Should NOT be exposed to the internet
- Has no authentication (trusted local network)

For remote access, use a VPN or SSH tunnel.

## License

GPL-3.0 - Same as the original Shake&Tune project.
