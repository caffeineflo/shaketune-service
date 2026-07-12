# Shake&Tune Service

A fork of [Klippain Shake&Tune](https://github.com/Frix-x/klippain-shaketune) that exposes graph generation through a Docker-based REST API. It moves numpy and matplotlib analysis off resource-constrained printers such as the Creality K1.

![logo banner](./docs/banner.png)

## Remote K1 workflow

The K1-side `SHAKETUNE_SHAPER_REMOTE` macro starts a detached runner and returns control to Klipper immediately. The runner refuses to move unless the printer is in `standby` with both heater targets at zero. It then processes one axis at a time:

1. Home and capture the X-axis resonance CSV.
2. Restart Klipper to clear the K1 Linux MCU move queue, then prove Klipper is `ready` and `homed_axes` was cleared.
3. Upload the X-axis CSV with the shared service token and verify the graph is retrievable.
4. Repeat the capture, restart, authenticated upload, and verification for the Y axis.

This keeps the two raw-data sweeps out of one Klipper command lifecycle. Shaper and belt runners share `/tmp/shaketune_calibration.lock`, so they can't move the same printer concurrently. Progress is written to `/tmp/shaketune_shaper.log`, and failed uploads keep their CSV and response evidence in `/tmp` for troubleshooting.

## Live validation

The production workflow completed two full X/Y shaper cycles per printer on 2026-07-12:

- K1v3: `20260712_133847` ([X](http://shaketune.iflorian.com:3080/results/k1v3/20260712_133847_shaper_x.png), [Y](http://shaketune.iflorian.com:3080/results/k1v3/20260712_133847_shaper_y.png)) and `20260712_141742` ([X](http://shaketune.iflorian.com:3080/results/k1v3/20260712_141742_shaper_x.png), [Y](http://shaketune.iflorian.com:3080/results/k1v3/20260712_141742_shaper_y.png))
- K1v4: `20260712_134753` ([X](http://shaketune.iflorian.com:3080/results/k1v4/20260712_134753_shaper_x.png), [Y](http://shaketune.iflorian.com:3080/results/k1v4/20260712_134753_shaper_y.png)) and `20260712_142621` ([X](http://shaketune.iflorian.com:3080/results/k1v4/20260712_142621_shaper_x.png), [Y](http://shaketune.iflorian.com:3080/results/k1v4/20260712_142621_shaper_y.png))

The final belt workflow also completed on [K1v3](http://shaketune.iflorian.com:3080/results/k1v3/20260712_144333_belts.png) and [K1v4](http://shaketune.iflorian.com:3080/results/k1v4/20260712_145249_belts.png). All eight shaper graphs are valid 2250x1740 PNGs, and both belt graphs are valid 2250x1050 PNGs. Each printer finished in `standby` with heater targets at zero, cleared raw files and locks, and recorded the expected firmware restarts.

This proves repeatability across those two shaper cycles and one belt cycle per printer. It isn't a guarantee that future hardware, firmware, network, or service changes can't fail.

## Authentication

All analysis endpoints require the shared token in the `X-ShakeTune-Token` request header. The service refuses to start without `SHAKETUNE_API_TOKEN_FILE` or `SHAKETUNE_API_TOKEN`. The file-based option is recommended so the token does not appear in Compose files or process environment listings.

Dashboard, result, latest-result, and health reads do not require the token. Keep those endpoints on a trusted network or behind an access-controlled reverse proxy.

Never commit the token to this repository.

## Quick start

Create a token file that is readable by the container's UID/GID 1000, then start the canonical Compose stack:

```bash
umask 077
mkdir -p secrets
openssl rand -hex 32 > secrets/api-token
sudo chown root:1000 secrets/api-token
sudo chmod 0440 secrets/api-token
sudo install -d -m 0770 -o 1000 -g 1000 /dockerfs/shaketune-results
docker compose up -d
curl --fail --silent --show-error http://127.0.0.1:3080/health
```

Copy the same token to each K1 without placing it in a command argument:

```bash
ssh CrealityK1v3 'umask 077; mkdir -p /usr/data/printer_data/config/shaketune; cat > /usr/data/printer_data/config/shaketune/token; chmod 600 /usr/data/printer_data/config/shaketune/token' < secrets/api-token
```

Then install the K1 macros and run:

```text
SHAKETUNE_SHAPER_REMOTE
```

The printer-specific result is available at `http://your-server:3080/latest/<printer-name>/shaper`.

See [service/README.md](./service/README.md) for the full service, multi-printer, installation, and troubleshooting guide.

## API

| Endpoint | Authentication | Description |
|----------|----------------|-------------|
| `POST /shaper` | `X-ShakeTune-Token` | Upload one or more axis CSVs and generate input shaper graphs |
| `POST /belts` | `X-ShakeTune-Token` | Upload two belt CSVs and generate a comparison graph |
| `POST /vibrations` | `X-ShakeTune-Token` | Upload a vibration CSV and generate a vibration graph |
| `GET /latest/{printer}/{type}` | None | Redirect to a printer's most recent graph of the requested type |
| `GET /results/{printer}/{file}` | None | Read a generated graph |
| `GET /health` | None | Read service health |

## Original documentation

For help interpreting the graphs, see the [Input Shaper Tuning Guide](./docs/is_tuning_generalities.md).

## License

GPL-3.0
