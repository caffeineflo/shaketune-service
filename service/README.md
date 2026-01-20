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
  -p 3080:8080 \
  -v /path/to/results:/app/results \
  ghcr.io/caffeineflo/shaketune-service:latest
```

### 2. Configure Your K1

Add the following to your Klipper config (e.g., in `improved-shapers.cfg` or a separate include file):

```ini
########################################
# Shake&Tune Service Integration
########################################

[gcode_shell_command shaketune_shaper]
command: sh /usr/data/printer_data/config/Helper-Script/improved-shapers/shaketune_upload.sh
timeout: 300.0
verbose: True

[gcode_shell_command shaketune_belts]
command: sh /usr/data/printer_data/config/Helper-Script/improved-shapers/shaketune_belts_upload.sh
timeout: 300.0
verbose: True

[gcode_macro SHAKETUNE_SHAPER]
description: Test X/Y resonances and generate graphs via remote Shake&Tune service
gcode:
  {% if printer.toolhead.homed_axes != "xyz" %}
    RESPOND TYPE=command MSG="Homing..."
    G28
  {% endif %}
  RESPOND TYPE=command MSG="Testing X Resonances..."
  TEST_RESONANCES AXIS=X OUTPUT=raw_data NAME=x
  M400
  RESPOND TYPE=command MSG="Testing Y Resonances..."
  TEST_RESONANCES AXIS=Y OUTPUT=raw_data NAME=y
  M400
  RESPOND TYPE=command MSG="Uploading to Shake&Tune service..."
  RESPOND TYPE=command MSG="Graph URL: http://YOUR_SERVER:3080/latest/shaper"
  RUN_SHELL_COMMAND CMD=shaketune_shaper
  RUN_SHELL_COMMAND CMD=delete_csv
  RESPOND TYPE=command MSG="Shake&Tune analysis complete!"

[gcode_macro SHAKETUNE_BELTS]
description: Test belt resonances and generate comparison graph via remote Shake&Tune service
gcode:
  {% set min_freq = params.FREQ_START|default(5)|float %}
  {% set max_freq = params.FREQ_END|default(133.33)|float %}
  {% set hz_per_sec = params.HZ_PER_SEC|default(1)|float %}
  {% if printer.toolhead.homed_axes != "xyz" %}
    RESPOND TYPE=command MSG="Homing..."
    G28
  {% endif %}
  RESPOND TYPE=command MSG="Testing Belt B (1,1)..."
  TEST_RESONANCES AXIS=1,1 OUTPUT=raw_data NAME=b FREQ_START={min_freq} FREQ_END={max_freq} HZ_PER_SEC={hz_per_sec}
  M400
  RESPOND TYPE=command MSG="Testing Belt A (1,-1)..."
  TEST_RESONANCES AXIS=1,-1 OUTPUT=raw_data NAME=a FREQ_START={min_freq} FREQ_END={max_freq} HZ_PER_SEC={hz_per_sec}
  M400
  RESPOND TYPE=command MSG="Uploading to Shake&Tune service..."
  RESPOND TYPE=command MSG="Graph URL: http://YOUR_SERVER:3080/latest/belts"
  RUN_SHELL_COMMAND CMD=shaketune_belts
  RUN_SHELL_COMMAND CMD=delete_csv
  RESPOND TYPE=command MSG="Shake&Tune belt analysis complete!"
```

### 3. Create Upload Scripts

Create the upload scripts on your K1. These handle the curl upload and display the result URL.

**`shaketune_upload.sh`:**
```bash
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
RESULT=$(curl POST "${SERVER}/shaper" \
  -F "files=@/tmp/raw_data_x_x.csv" \
  -F "files=@/tmp/raw_data_y_y.csv" 2>&1)

URL=$(echo "$RESULT" | sed -n 's/.*"url":"\([^"]*\)".*/\1/p')

if [ -n "$URL" ]; then
  FULL_URL="${SERVER}${URL}"
  echo "==================================="
  echo "Shake&Tune Graph Ready!"
  echo "URL: ${FULL_URL}"
  echo "==================================="
  echo "$FULL_URL" > /tmp/shaketune_graph_url.txt
else
  echo "Error: $RESULT"
fi
```

**`shaketune_belts_upload.sh`:**
```bash
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
RESULT=$(curl POST "${SERVER}/belts" \
  -F "files=@/tmp/raw_data_axis=1.000,-1.000_a.csv" \
  -F "files=@/tmp/raw_data_axis=1.000,1.000_b.csv" 2>&1)

URL=$(echo "$RESULT" | sed -n 's/.*"url":"\([^"]*\)".*/\1/p')

if [ -n "$URL" ]; then
  FULL_URL="${SERVER}${URL}"
  echo "==================================="
  echo "Shake&Tune Belts Graph Ready!"
  echo "URL: ${FULL_URL}"
  echo "==================================="
  echo "$FULL_URL" > /tmp/shaketune_belts_url.txt
else
  echo "Error: $RESULT"
fi
```

> **Note:** K1 uses BusyBox curl which has different syntax than standard curl. Use `curl POST url` instead of `curl -X POST url`.

### 4. Run Calibration

From Mainsail/Fluidd console:

```gcode
SHAKETUNE_SHAPER    # Input shaper calibration
SHAKETUNE_BELTS     # Belt tension comparison
```

The console will display a predictable URL (`/latest/shaper` or `/latest/belts`) immediately. This URL always redirects to the most recent graph, so you can open it right away and refresh once processing completes.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shaper` | POST | Upload resonance CSVs, get shaper graph |
| `/belts` | POST | Upload belt test CSVs, get comparison graph |
| `/vibrations` | POST | Upload vibration CSVs, get analysis graph |
| `/results/{file}` | GET | Retrieve generated graph by filename |
| `/latest/shaper` | GET | Redirect to most recent shaper graph |
| `/latest/belts` | GET | Redirect to most recent belts graph |
| `/latest/vibrations` | GET | Redirect to most recent vibrations graph |
| `/health` | GET | Health check |
| `/` | GET | API documentation |

### Example API Usage

```bash
# Analyze shaper resonances
curl -X POST http://localhost:3080/shaper \
  -F "files=@raw_data_x_x.csv" \
  -F "files=@raw_data_y_y.csv"

# Response:
# {"url": "/results/20241204_123456_abc123_shaper.png", "id": "abc123", "type": "shaper"}

# Get the latest shaper graph (redirects to most recent)
curl -L http://localhost:3080/latest/shaper -o latest_shaper.png
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
docker run -p 3080:8080 -v $(pwd)/results:/app/results shaketune-service
```

## Docker Compose Example

```yaml
services:
  shaketune:
    image: ghcr.io/caffeineflo/shaketune-service:latest
    pull_policy: always
    container_name: shaketune-service
    ports:
      - "3080:8080"
    volumes:
      - /path/to/results:/app/results
    restart: unless-stopped
```

## Persistent Storage

Results are stored in `/app/results` inside the container. Mount a volume or bind mount to persist graphs:

```yaml
volumes:
  - shaketune-results:/app/results  # Docker volume
  # OR
  - /dockerfs/shaketune-results:/app/results  # Bind mount
```

## Network Security

The service is designed for local network use only. It:
- Binds to all interfaces (0.0.0.0) within the container
- Should NOT be exposed to the internet
- Has no authentication (trusted local network)

For remote access, use a VPN or SSH tunnel.

## Troubleshooting

### K1 crashes during resonance test
The K1 has limited RAM. If running both X and Y tests causes "Move queue overflow" errors, try:
- Running tests separately with a firmware restart between them
- Using the background upload script to avoid blocking Klipper during upload

### "No valid data found" error
Ensure your CSV files contain valid accelerometer data. The files should be generated by `TEST_RESONANCES ... OUTPUT=raw_data`.

### Graph dimension mismatch errors
This was fixed in recent versions. Make sure you're running the latest image: `docker pull ghcr.io/caffeineflo/shaketune-service:latest`

## License

GPL-3.0 - Same as the original Shake&Tune project.
