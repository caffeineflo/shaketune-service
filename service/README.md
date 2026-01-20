# Shake&Tune Analysis Service

A Docker-based service that processes Klipper accelerometer data and generates input shaper graphs. Offloads heavy numpy/matplotlib processing from resource-constrained printers like the Creality K1.

## Running the Service

```bash
docker run -d --name shaketune-service \
  -p 3080:8080 \
  -v /path/to/results:/app/results \
  ghcr.io/caffeineflo/shaketune-service:latest
```

Or with Docker Compose:

```yaml
services:
  shaketune:
    image: ghcr.io/caffeineflo/shaketune-service:latest
    container_name: shaketune-service
    ports:
      - "3080:8080"
    volumes:
      - /dockerfs/shaketune-results:/app/results
    restart: unless-stopped
```

## K1 Integration

### 1. Create upload scripts

These scripts generate a timestamp, show the predictable URL immediately, then upload.

**`/usr/data/printer_data/config/shaketune_upload.sh`:**
```sh
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
PRINTER="k1v4"  # Change per printer
TS=$(date +%Y%m%d_%H%M%S)
echo "==========================================="
echo "Graph URL: ${SERVER}/results/${PRINTER}/${TS}_shaper.png"
echo "==========================================="
curl POST "${SERVER}/shaper" \
  -F "files=@/tmp/raw_data_x_x.csv" \
  -F "files=@/tmp/raw_data_y_y.csv" \
  -F "printer=${PRINTER}" \
  -F "timestamp=${TS}"
```

**`/usr/data/printer_data/config/shaketune_belts_upload.sh`:**
```sh
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
PRINTER="k1v4"  # Change per printer
TS=$(date +%Y%m%d_%H%M%S)
echo "==========================================="
echo "Graph URL: ${SERVER}/results/${PRINTER}/${TS}_belts.png"
echo "==========================================="
curl POST "${SERVER}/belts" \
  -F "files=@/tmp/raw_data_axis=1.000,-1.000_a.csv" \
  -F "files=@/tmp/raw_data_axis=1.000,1.000_b.csv" \
  -F "printer=${PRINTER}" \
  -F "timestamp=${TS}"
```

Make executable: `chmod +x /usr/data/printer_data/config/shaketune_*.sh`

### 2. Add to Klipper config

```ini
[gcode_shell_command shaketune_shaper]
command: sh /usr/data/printer_data/config/shaketune_upload.sh
timeout: 300.0
verbose: True

[gcode_shell_command shaketune_belts]
command: sh /usr/data/printer_data/config/shaketune_belts_upload.sh
timeout: 300.0
verbose: True

[gcode_macro SHAKETUNE_SHAPER]
description: Test X/Y resonances via remote Shake&Tune service
gcode:
  {% if printer.toolhead.homed_axes != "xyz" %}
    G28
  {% endif %}
  RESPOND MSG="Testing X axis..."
  TEST_RESONANCES AXIS=X OUTPUT=raw_data NAME=x
  M400
  RESPOND MSG="Testing Y axis..."
  TEST_RESONANCES AXIS=Y OUTPUT=raw_data NAME=y
  M400
  RESPOND MSG="Uploading..."
  RUN_SHELL_COMMAND CMD=shaketune_shaper

[gcode_macro SHAKETUNE_BELTS]
description: Test belt resonances via remote Shake&Tune service
gcode:
  {% if printer.toolhead.homed_axes != "xyz" %}
    G28
  {% endif %}
  RESPOND MSG="Testing belt B..."
  TEST_RESONANCES AXIS=1,1 OUTPUT=raw_data NAME=b
  M400
  RESPOND MSG="Testing belt A..."
  TEST_RESONANCES AXIS=1,-1 OUTPUT=raw_data NAME=a
  M400
  RESPOND MSG="Uploading..."
  RUN_SHELL_COMMAND CMD=shaketune_belts
```

### 3. Restart Klipper and run

```
SHAKETUNE_SHAPER
SHAKETUNE_BELTS
```

The graph URL is shown immediately when upload starts - no need to wait for processing.

## Multi-Printer Setup

Each printer gets its own subdirectory in results:
- `http://server:3080/results/k1v3/20260120_120000_shaper.png`
- `http://server:3080/results/k1v4/20260120_120000_shaper.png`

Just set a unique `PRINTER=` value in each printer's upload scripts.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shaper` | POST | Upload X/Y CSVs → shaper graph |
| `/belts` | POST | Upload belt CSVs → comparison graph |
| `/vibrations` | POST | Upload vibration CSVs → analysis graph |
| `/results/{printer}/{file}` | GET | Get specific graph |
| `/latest/{printer}/{type}` | GET | Redirect to printer's latest graph |
| `/latest/{type}` | GET | Redirect to latest (default printer) |
| `/health` | GET | Health check |

### Parameters

All POST endpoints accept:
- `files` (required) - CSV files from TEST_RESONANCES
- `printer` (optional) - Printer name for organization, default: "default"
- `timestamp` (optional) - Client timestamp for predictable URLs, format: YYYYMMDD_HHMMSS

## Troubleshooting

**"Move queue overflow" during test:** K1 RAM limitation. Run X and Y tests separately with `FIRMWARE_RESTART` between them.

**No graph generated:** Check logs: `docker logs shaketune-service`

**BusyBox curl:** K1 uses `curl POST url` not `curl -X POST url`.

## License

GPL-3.0
