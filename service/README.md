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

### 1. Add to your Klipper config

Add this to `printer.cfg` or an included config file (e.g., `shaketune.cfg`):

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
  RESPOND MSG="Graph: http://YOUR_SERVER:3080/latest/shaper"
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
  RESPOND MSG="Graph: http://YOUR_SERVER:3080/latest/belts"
  RUN_SHELL_COMMAND CMD=shaketune_belts
```

### 2. Create upload scripts

**`/usr/data/printer_data/config/shaketune_upload.sh`:**
```sh
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
curl POST "${SERVER}/shaper" \
  -F "files=@/tmp/raw_data_x_x.csv" \
  -F "files=@/tmp/raw_data_y_y.csv"
```

**`/usr/data/printer_data/config/shaketune_belts_upload.sh`:**
```sh
#!/bin/sh
SERVER="http://YOUR_SERVER:3080"
curl POST "${SERVER}/belts" \
  -F "files=@/tmp/raw_data_axis=1.000,-1.000_a.csv" \
  -F "files=@/tmp/raw_data_axis=1.000,1.000_b.csv"
```

Make them executable: `chmod +x /usr/data/printer_data/config/shaketune_*.sh`

> **Note:** K1 uses BusyBox curl - use `curl POST url` not `curl -X POST url`.

### 3. Replace `YOUR_SERVER` with your Docker host IP

### 4. Restart Klipper and run

```
SHAKETUNE_SHAPER
SHAKETUNE_BELTS
```

Graph URL is shown immediately - `/latest/shaper` always redirects to the most recent result.

## API

| Endpoint | Description |
|----------|-------------|
| `POST /shaper` | Upload X/Y CSVs → shaper graph |
| `POST /belts` | Upload belt CSVs → comparison graph |
| `GET /latest/shaper` | Redirect to latest shaper graph |
| `GET /latest/belts` | Redirect to latest belts graph |
| `GET /results/{file}` | Get specific graph |
| `GET /health` | Health check |

## Troubleshooting

**"Move queue overflow" during test:** K1 RAM limitation. Try running X and Y tests separately with `FIRMWARE_RESTART` between them.

**No graph generated:** Check container logs: `docker logs shaketune-service`

## License

GPL-3.0
