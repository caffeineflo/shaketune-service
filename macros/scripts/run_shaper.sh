#!/bin/sh
# Run one input-shaper axis at a time and upload each result to the remote
# service. Firmware restarts between sweeps keep the K1 Linux MCU healthy.

set -eu

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
UPLOAD_URL="${BASE_URL%/}"
FREQ_START="${4:-5}"
FREQ_END="${5:-133.33}"
HZ_PER_SEC="${6:-1}"
TOKEN_FILE="${SHAKETUNE_TOKEN_FILE:-/usr/data/printer_data/config/shaketune/token}"
LOCK_DIR="${SHAKETUNE_LOCK_DIR:-/tmp/shaketune_calibration.lock}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: Another ShakeTune calibration workflow is already running"
  exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT HUP INT TERM

if [ ! -r "$TOKEN_FILE" ]; then
  echo "ERROR: ShakeTune token is missing or unreadable: $TOKEN_FILE"
  exit 1
fi
TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
if [ -z "$TOKEN" ]; then
  echo "ERROR: ShakeTune token is empty: $TOKEN_FILE"
  exit 1
fi

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
}

moonraker_get() {
  wget -q -O - -T 10 "http://localhost:7125/$1" 2>/dev/null
}

send_gcode() {
  encoded="$(printf '%s' "$1" | sed 's/%/%25/g; s/ /%20/g; s/"/%22/g; s/=/%3D/g; s/,/%2C/g')"
  if ! wget -q -O /dev/null -T 360 "http://localhost:7125/printer/gcode/script?script=${encoded}" 2>/dev/null; then
    echo "ERROR: Moonraker rejected or timed out while running: $1"
    return 1
  fi
}

numeric_zero() {
  awk -v value="$1" 'BEGIN { exit !(value == 0) }'
}

require_idle_and_cold() {
  print_json="$(moonraker_get 'printer/objects/query?print_stats')" || {
    echo "ERROR: Cannot read print state from Moonraker"
    return 1
  }
  case "$print_json" in
    *'"state": "standby"'*|*'"state":"standby"'*) ;;
    *)
      echo "ERROR: Printer is not in standby; refusing calibration motion"
      return 1
      ;;
  esac

  for heater in extruder heater_bed; do
    heater_json="$(moonraker_get "printer/objects/query?${heater}")" || {
      echo "ERROR: Cannot read ${heater} state from Moonraker"
      return 1
    }
    target="$(printf '%s\n' "$heater_json" | sed -n 's/.*"target": \([-0-9.]*\).*/\1/p')"
    if [ -z "$target" ] || ! numeric_zero "$target"; then
      echo "ERROR: ${heater} target is ${target:-unknown}; refusing calibration motion"
      return 1
    fi
  done
}

wait_ready() {
  elapsed=0
  while [ "$elapsed" -lt 90 ]; do
    info="$(moonraker_get 'printer/info' || true)"
    case "$info" in
      *'"state": "ready"'*|*'"state":"ready"'*)
        require_idle_and_cold
        return $?
        ;;
    esac
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "ERROR: Klipper did not become ready within 90 seconds"
  return 1
}

wait_homed() {
  elapsed=0
  while [ "$elapsed" -lt 90 ]; do
    state="$(moonraker_get 'printer/objects/query?toolhead' || true)"
    case "$state" in
      *'"homed_axes": "xyz"'*|*'"homed_axes":"xyz"'*) return 0 ;;
    esac
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "ERROR: Printer did not finish homing within 90 seconds"
  return 1
}

wait_for_csv() {
  target_file="$1"
  elapsed=0
  previous_size=0
  stable_count=0

  while [ "$elapsed" -lt 300 ]; do
    if [ -f "$target_file" ]; then
      current_size="$(wc -c < "$target_file" 2>/dev/null || echo 0)"
      if [ "$current_size" -gt 1000000 ] && [ "$current_size" -eq "$previous_size" ]; then
        stable_count=$((stable_count + 1))
        if [ "$stable_count" -ge 2 ]; then
          echo "$current_size"
          return 0
        fi
      else
        stable_count=0
      fi
      previous_size="$current_size"
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done

  echo "ERROR: Resonance CSV did not stabilize within 300 seconds" >&2
  return 1
}

firmware_restart() {
  log "Restarting Klipper to clear the K1 move queue"
  sh "${SCRIPT_DIR}/firmware_restart.sh"
}

upload_axis() {
  axis="$1"
  csv_file="$2"
  gzip_file="${csv_file}.gz"
  response_file="/tmp/shaketune_shaper_${axis}_response.json"
  error_file="/tmp/shaketune_shaper_${axis}_upload.err"
  result_url="${BASE_URL}/results/${PRINTER}/${TIMESTAMP}_shaper_${axis}.png"

  log "Compressing ${axis}-axis data"
  nice -n 19 gzip -f -k "$csv_file"

  log "Uploading ${axis}-axis data"
  if ! curl --timeout 360 -H "X-ShakeTune-Token: ${TOKEN}" POST "${UPLOAD_URL}/shaper" \
    -F "files=@${gzip_file}" \
    -F "printer=${PRINTER}" \
    -F "timestamp=${TIMESTAMP}" > "$response_file" 2> "$error_file"; then
    echo "ERROR: Upload transport failed for ${axis} axis"
    sed -n '1,10p' "$error_file"
    return 1
  fi

  compact_response="$(tr -d '[:space:]' < "$response_file")"
  case "$compact_response" in
    *'"axis":"'"${axis}"'"'*) ;;
    *)
      echo "ERROR: Service did not confirm the ${axis}-axis result"
      sed -n '1,20p' "$response_file"
      return 1
      ;;
  esac
  case "$compact_response" in
    *'"url":"/results/'"${PRINTER}"'/'"${TIMESTAMP}"'_shaper_'"${axis}"'.png"'*) ;;
    *)
      echo "ERROR: Service returned an unexpected ${axis}-axis graph URL"
      sed -n '1,20p' "$response_file"
      return 1
      ;;
  esac

  if ! wget -q -O /dev/null -T 30 "$result_url"; then
    echo "ERROR: Generated ${axis}-axis graph is not retrievable: $result_url"
    return 1
  fi

  rm -f "$csv_file" "$gzip_file" "$response_file" "$error_file"
  log "Verified ${axis}-axis graph: $result_url"
}

run_axis() {
  axis="$1"
  csv_file="/tmp/raw_data_${axis}_${axis}.csv"

  require_idle_and_cold
  rm -f "$csv_file" "${csv_file}.gz"
  log "Homing before ${axis}-axis sweep"
  send_gcode 'G28'
  wait_homed
  send_gcode 'G0 X110 Y110 Z50 F6000'
  send_gcode 'M400'
  log "Starting ${axis}-axis resonance sweep"
  send_gcode "TEST_RESONANCES AXIS=${axis} OUTPUT=raw_data NAME=${axis} FREQ_START=${FREQ_START} FREQ_END=${FREQ_END} HZ_PER_SEC=${HZ_PER_SEC}"
  size="$(wait_for_csv "$csv_file")"
  log "${axis}-axis sweep complete: ${size} bytes"
  firmware_restart
  upload_axis "$axis" "$csv_file"
}

log "Starting authenticated input-shaper workflow for ${PRINTER}"
wait_ready
run_axis x
run_axis y
send_gcode 'RESPOND MSG="Remote ShakeTune input shaper complete"' || true
log "Input-shaper workflow complete"
