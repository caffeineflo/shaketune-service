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

## K1 Installation

SSH into your K1 and run the installer. You need the `macros/` directory from this repo (clone it or copy it to the K1 first):

```bash
# On the K1 via SSH
cd /tmp
wget -O install.tar.gz https://github.com/caffeineflo/shaketune-service/archive/refs/heads/main.tar.gz
tar xzf install.tar.gz --strip-components=2 shaketune-service-main/macros
sh install.sh shaketune.iflorian.com 3080 k1v3 http://shaketune.iflorian.com:3080
```

The installer:
1. Copies scripts to `/usr/data/printer_data/config/shaketune/scripts/`
2. Generates `shaketune.cfg` with your server host, port, and printer name baked in
3. Adds `[include shaketune.cfg]` to `printer.cfg` if not already present

After installing, do a `FIRMWARE_RESTART` in the Klipper console.

## Usage

### Input Shaper Calibration
```
SHAKETUNE_SHAPER_REMOTE
```
Homes, centers toolhead, runs X and Y resonance tests, uploads CSVs to service, returns graph URL. Takes ~4 minutes.

### Belt Comparison
```
SHAKETUNE_BELTS_REMOTE
```
Runs Belt A test, does a firmware restart (to avoid K1 move queue overflow with raw data), runs Belt B test, uploads both. Takes ~6 minutes. Progress logged to `/tmp/shaketune_belts.log`.

**Why the firmware restart?** The K1 has 209MB RAM. Two consecutive `TEST_RESONANCES` with `OUTPUT=raw_data` generates ~20MB per sweep. The move queue overflows before the second sweep completes, causing an MCU shutdown. The firmware restart between sweeps clears the queue. The script survives the restart because it runs detached from Klipper via `setsid`.

### Excite at Frequency
```
SHAKETUNE_EXCITE_REMOTE FREQUENCY=50 DURATION=10 AXIS=x
```
Vibrates at a specific frequency for a set duration to locate resonance sources. Useful for diagnosing where a specific frequency peak is coming from.

## Multi-Printer Setup

Run the installer on each K1 with a unique printer name:
```bash
# On K1v3
sh install.sh shaketune.iflorian.com 3080 k1v3 http://shaketune.iflorian.com:3080

# On K1v4
sh install.sh shaketune.iflorian.com 3080 k1v4 http://shaketune.iflorian.com:3080
```

Each printer gets its own subdirectory in results:
- `http://server:3080/results/k1v3/20260120_120000_shaper.png`
- `http://server:3080/results/k1v4/20260120_120000_shaper.png`

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shaper` | POST | Upload X/Y CSVs, get input shaper graph |
| `/belts` | POST | Upload belt CSVs, get comparison graph |
| `/vibrations` | POST | Upload vibration CSVs, get analysis graph |
| `/results/{printer}/{file}` | GET | Get specific graph |
| `/latest/{printer}/{type}` | GET | Redirect to printer's latest graph |
| `/latest/{type}` | GET | Redirect to latest (default printer) |
| `/health` | GET | Health check |
| `/` | GET | Web dashboard |

### Parameters

All POST endpoints accept:
- `files` (standard curl) or `file_x`/`file_y`, `file_a`/`file_b` (BusyBox curl)
- `printer` (optional) - Printer name, default: "default"
- `timestamp` (optional) - Client timestamp for predictable URLs, format: YYYYMMDD_HHMMSS

Files can be gzipped (`.csv.gz`) and will be automatically decompressed.

## Troubleshooting

**"Move queue overflow" during belt test:** This is handled automatically by `SHAKETUNE_BELTS_REMOTE`. If you see it anyway, check `/tmp/shaketune_belts.log` for details.

**Belt test seems stuck:** Check progress with `cat /tmp/shaketune_belts.log` via SSH. The test takes ~6 minutes total.

**No graph generated:** Check service logs: `docker logs shaketune-service`

**BusyBox curl:** K1 uses `curl POST url` not `curl -X POST url`. The scripts handle this.

## License

GPL-3.0
