# Shake&Tune Analysis Service

A Docker service that processes Klipper accelerometer data and generates input shaper, belt comparison, and vibration graphs. It offloads the heavy numpy and matplotlib work from resource-constrained printers such as the Creality K1.

## Security model

The service requires a shared token for all analysis requests:

- `POST /shaper`
- `POST /belts`
- `POST /vibrations`

Clients send the token in the `X-ShakeTune-Token` header. Configure the service with exactly the same value using one of these settings:

- `SHAKETUNE_API_TOKEN_FILE`: recommended for deployments
- `SHAKETUNE_API_TOKEN`: suitable for local testing only

The service fails closed at startup when neither setting contains a non-empty token. Its dashboard, graph results, latest-result redirects, and `/health` endpoint remain readable without the token. Restrict those read endpoints to a trusted LAN or an access-controlled reverse proxy.

Never commit a service token. Give the token file only the permissions needed by the container's UID/GID 1000.

## Run with Docker Compose

The repository's [`docker-compose.yml`](../docker-compose.yml) is the canonical deployment definition. Create the bind-mounted token and start the service:

```bash
cd /path/to/shaketune-service
umask 077
mkdir -p secrets
openssl rand -hex 32 > secrets/api-token
sudo chown root:1000 secrets/api-token
sudo chmod 0440 secrets/api-token
sudo install -d -m 0770 -o 1000 -g 1000 /dockerfs/shaketune-results
docker compose up -d
curl --fail --silent --show-error http://127.0.0.1:3080/health
```

The Compose stack mounts `secrets/api-token` at `/run/secrets/shaketune_api_token` and configures `SHAKETUNE_API_TOKEN_FILE` accordingly.

For a direct `docker run`, create a results directory writable by UID/GID 1000 and mount the same token file read-only:

```bash
sudo install -d -m 0770 -o 1000 -g 1000 /path/to/results
docker run -d --name shaketune-service \
  -p 3080:8080 \
  -v /path/to/results:/app/results \
  -v /path/to/api-token:/run/secrets/shaketune_api_token:ro \
  -e SHAKETUNE_API_TOKEN_FILE=/run/secrets/shaketune_api_token \
  ghcr.io/caffeineflo/shaketune-service:latest
```

## K1 installation

The K1 runner reads the token from `/usr/data/printer_data/config/shaketune/token`. Copy the service token over standard input so it is not exposed as a command argument:

```bash
ssh CrealityK1v3 'umask 077; mkdir -p /usr/data/printer_data/config/shaketune; cat > /usr/data/printer_data/config/shaketune/token; chmod 600 /usr/data/printer_data/config/shaketune/token' < /path/to/api-token
```

Download the repository's `macros/` directory on the printer and run the installer:

```bash
cd /tmp
wget -O install.tar.gz https://github.com/caffeineflo/shaketune-service/archive/refs/heads/main.tar.gz
tar xzf install.tar.gz --strip-components=2 shaketune-service-main/macros
sh install.sh shaketune.iflorian.com 3080 k1v3 http://shaketune.iflorian.com:3080
```

The installer:

1. Requires the non-empty token file and sets it to mode `0600`.
2. Copies the upload and detached runner scripts to `/usr/data/printer_data/config/shaketune/scripts/`.
3. Generates `shaketune.cfg` with the server and printer settings.
4. Adds `[include shaketune.cfg]` to `printer.cfg` if it is not already present.

Run `FIRMWARE_RESTART` in the Klipper console after installation.

## Input shaper workflow

Start an authenticated remote calibration from the Klipper console:

```text
SHAKETUNE_SHAPER_REMOTE
```

The macro performs a fast safety check, launches `run_shaper.sh` through `setsid`, and returns without waiting for the long resonance workflow. The detached runner independently verifies that Klipper is ready, the printer is in `standby`, and both heater targets are zero before every motion sequence. It also shares `/tmp/shaketune_calibration.lock` with the belt runner, which prevents the two calibration workflows from overlapping.

For each axis, the runner homes the printer, centers the toolhead, captures one raw-data sweep, and requests a firmware restart. It continues only after Klipper returns to `ready` with `homed_axes` cleared, then sends the compressed CSV with `X-ShakeTune-Token` and verifies the exact graph URL before removing local evidence. The X axis must finish successfully before Y begins. This avoids the MCU shutdown seen when both raw-data sweeps remain in a single Klipper command lifecycle and prevents a rejected restart from looking successful.

Follow progress over SSH:

```bash
tail -f /tmp/shaketune_shaper.log
```

On failure, the runner exits before the next axis and preserves relevant files such as these:

- `/tmp/raw_data_x_x.csv`
- `/tmp/raw_data_x_x.csv.gz`
- `/tmp/shaketune_shaper_x_response.json`
- `/tmp/shaketune_shaper_x_upload.err`

## Other macros

### Belt comparison

```text
SHAKETUNE_BELTS_REMOTE
```

The detached belt runner captures Belt A, verifies a firmware restart, captures Belt B, and verifies a final restart before its authenticated upload. It shares the shaper lock and removes raw CSVs only after the service response and generated graph are verified. Progress is logged to `/tmp/shaketune_belts.log`.

### Excite at a frequency

```text
SHAKETUNE_EXCITE_REMOTE FREQUENCY=50 DURATION=10 AXIS=x
```

This moves the selected axis around a narrow frequency range to help locate a resonance source. It does not upload analysis data.

## Multi-printer setup

Provision the same token on every printer, but give each printer a stable, unique name:

```bash
# On K1v3
sh install.sh shaketune.iflorian.com 3080 k1v3 http://shaketune.iflorian.com:3080

# On K1v4
sh install.sh shaketune.iflorian.com 3080 k1v4 http://shaketune.iflorian.com:3080
```

Each result is stored in its printer directory. Because the shaper runner uploads one axis at a time, a completed calibration produces two independently verified graphs:

- `http://server:3080/results/k1v3/20260120_120000_shaper_x.png`
- `http://server:3080/results/k1v3/20260120_120000_shaper_y.png`

The latest result redirect is `http://server:3080/latest/k1v3/shaper`.

## API

| Endpoint | Method | Authentication | Description |
|----------|--------|----------------|-------------|
| `/shaper` | POST | Required | Upload axis CSVs and generate input shaper graphs |
| `/belts` | POST | Required | Upload exactly two belt CSVs and generate a comparison graph |
| `/vibrations` | POST | Required | Upload a vibration CSV and generate a vibration graph |
| `/results/{printer}/{file}` | GET | None | Read a specific graph |
| `/latest/{printer}/{type}` | GET | None | Redirect to a printer's latest graph |
| `/latest/{type}` | GET | None | Redirect to the default printer's latest graph |
| `/health` | GET | None | Read service health |
| `/` | GET | None | Read the web dashboard |

All POST endpoints accept multipart form data with these common fields:

- `files` or the endpoint-specific BusyBox fields such as `file_x` and `file_y`
- `printer`: a stable printer name containing letters, numbers, underscores, or hyphens
- `timestamp`: an optional client timestamp in `YYYYMMDD_HHMMSS` format

The `/belts` endpoint also accepts `kinematics`, which defaults to `corexy`. Passing the kinematics through to Shake&Tune enables its similarity and experimental mechanical-health calculations for supported CoreXY and CoreXZ variants.

Upload files must use `.csv` or `.csv.gz`. The service caps the aggregate incoming request body at 64 MiB by default, including multipart framing and chunked requests, before temporary multipart storage can exceed that limit. Set `SHAKETUNE_MAX_REQUEST_BODY_BYTES` to a positive byte count to change the cap. It also enforces per-file compressed and expanded size limits and serializes heavy analysis by default.

The service analysis timeout is 300 seconds by default. Official K1 upload scripts allow 360 seconds so a valid server-side analysis can finish before the client gives up.

## Troubleshooting

### The macro refuses to start

The printer must report `standby`, and the extruder and bed targets must both be zero. This is intentional: calibration motion must never start during a print or active heat-up.

If another shaper or belt job already owns `/tmp/shaketune_calibration.lock`, the runner exits before any motion. Don't run `SHAKETUNE_EXCITE_REMOTE` while a detached calibration is active; that inline diagnostic doesn't use the shell-workflow lock.

### The service returns 401

The printer token is missing or does not exactly match the service token. Check that `/usr/data/printer_data/config/shaketune/token` is non-empty and mode `0600`, then compare it with the deployed service secret without printing either value into logs.

### A shaper run stops after one axis

Read `/tmp/shaketune_shaper.log`, the saved response JSON, and the upload error file. The runner intentionally does not start Y when X capture, restart, upload, or graph verification fails.

### Move queue or Linux MCU shutdown

Confirm that `SHAKETUNE_SHAPER_REMOTE` points to the detached `run_shaper.sh` workflow. A synchronous macro that runs both `TEST_RESONANCES` commands back-to-back can exhaust the K1's small move queue.

Each complete shaper or belt log must contain two `Firmware restart verified` lines. Missing verification is a failed workflow even if Klipper still answers requests.

### No graph appears

Check the service health and logs:

```bash
curl --fail --silent --show-error http://server:3080/health
docker logs shaketune-service
```

The K1 uses BusyBox curl syntax (`curl POST URL`) rather than the desktop curl `-X POST` form. The installed scripts handle this difference and verify the generated result with `wget`.

## License

GPL-3.0
