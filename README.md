# Shake&Tune Service

A fork of [Klippain Shake&Tune](https://github.com/Frix-x/klippain-shaketune) modified to run as a standalone Docker-based REST API service for offloading input shaper analysis from resource-constrained printers like the Creality K1.

![logo banner](./docs/banner.png)

## Why This Fork?

The original Shake&Tune plugin runs analysis directly on the printer. This works great on powerful hardware, but the K1's MIPS-based CPU struggles with the heavy numpy/matplotlib processing required for graph generation.

This fork adds a **REST API service** that:
- Runs on any Docker host (NAS, Raspberry Pi 4, desktop, etc.)
- Accepts accelerometer CSV data via HTTP
- Returns URLs to generated graphs
- Keeps the lightweight data collection on the printer

## Quick Start

See the **[Service Documentation](./service/README.md)** for complete setup instructions.

### TL;DR

```bash
# 1. Run the service on your Docker host
docker run -d --name shaketune-service \
  -p 3080:8080 \
  -v /path/to/results:/app/results \
  ghcr.io/caffeineflo/shaketune-service:latest

# 2. Add macros to your K1 (see service/README.md)

# 3. Run from Klipper console
SHAKETUNE_SHAPER
# Graph URL: http://your-server:3080/latest/shaper
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/shaper` | POST | Upload resonance CSVs, get shaper graph |
| `/belts` | POST | Upload belt test CSVs, get comparison graph |
| `/vibrations` | POST | Upload vibration CSVs, get analysis graph |
| `/latest/{type}` | GET | Redirect to most recent graph |
| `/results/{file}` | GET | Retrieve graph by filename |
| `/health` | GET | Health check |

## Original Shake&Tune Documentation

For understanding the graphs and tuning methodology, see the original documentation:
- **[Input Shaper Tuning Guide](./docs/is_tuning_generalities.md)**
- **[Macro Documentation](./docs/macros/)**
- **[CLI Usage](./docs/cli_usage.md)**

## Credits

- Original [Klippain Shake&Tune](https://github.com/Frix-x/klippain-shaketune) by [Frix-x](https://github.com/Frix-x)
- Service adaptation for K1 by [caffeineflo](https://github.com/caffeineflo)

## License

GPL-3.0 - Same as the original Shake&Tune project.
