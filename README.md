# Shake&Tune Service

A fork of [Klippain Shake&Tune](https://github.com/Frix-x/klippain-shaketune) that runs as a Docker-based REST API. Offloads input shaper analysis from resource-constrained printers like the Creality K1.

![logo banner](./docs/banner.png)

## Quick Start

```bash
# Run the service
docker run -d -p 3080:8080 -v ./results:/app/results \
  ghcr.io/caffeineflo/shaketune-service:latest

# From your K1
SHAKETUNE_SHAPER
# → Graph: http://your-server:3080/latest/shaper
```

**[Full setup instructions →](./service/README.md)**

## API

| Endpoint | Description |
|----------|-------------|
| `POST /shaper` | Upload resonance CSVs → input shaper graph |
| `POST /belts` | Upload belt CSVs → comparison graph |
| `GET /latest/{type}` | Redirect to most recent graph |

## Original Documentation

For understanding the graphs: **[Input Shaper Tuning Guide](./docs/is_tuning_generalities.md)**

## License

GPL-3.0
