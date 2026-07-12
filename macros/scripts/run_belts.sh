#!/bin/sh
# Run belt comparison test with firmware restart between sweeps to avoid
# "Move queue overflow" on K1's 209MB RAM.
#
# This script orchestrates the full sequence via Moonraker API:
#   1. Home + center + run Belt A
#   2. Firmware restart (clears move queue)
#   3. Wait for Klipper ready
#   4. Home + center + run Belt B
#   5. Firmware restart (leaves the printer ready with a clear move queue)
#   6. Upload both CSVs to shaketune service
#
# Must be launched detached from Klipper (via setsid) because the
# firmware restart would kill a Klipper-owned child process.
#
# Usage: run_belts.sh HOST PORT PRINTER [FREQ_START] [FREQ_END] [HZ_PER_SEC]

set -eu

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
FREQ_START="${4:-5}"
FREQ_END="${5:-133.33}"
HZ_PER_SEC="${6:-1}"

FILE_A="${SHAKETUNE_BELT_FILE_A:-/tmp/raw_data_axis=1.000,-1.000_a.csv}"
FILE_B="${SHAKETUNE_BELT_FILE_B:-/tmp/raw_data_axis=1.000,1.000_b.csv}"
TOKEN_FILE="${SHAKETUNE_TOKEN_FILE:-/usr/data/printer_data/config/shaketune/token}"
LOCK_DIR="${SHAKETUNE_LOCK_DIR:-/tmp/shaketune_calibration.lock}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOONRAKER_URL="${MOONRAKER_URL:-http://localhost:7125}"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: Another ShakeTune calibration workflow is already running"
  exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT HUP INT TERM

moonraker_get() {
  wget -q -O - -T 10 "${MOONRAKER_URL%/}/$1" 2>/dev/null
}

# Helper: send gcode via Moonraker
send_gcode() {
  ENCODED=$(printf '%s' "$1" | sed 's/%/%25/g; s/ /%20/g; s/"/%22/g; s/=/%3D/g; s/,/%2C/g')
  if ! wget -q -O /dev/null -T 360 "${MOONRAKER_URL%/}/printer/gcode/script?script=${ENCODED}" 2>/dev/null; then
    echo "ERROR: Moonraker rejected or timed out while running: $1"
    return 1
  fi
}

firmware_restart() {
  echo "Restarting Klipper to clear the K1 move queue"
  sh "${SCRIPT_DIR}/firmware_restart.sh"
}

numeric_zero() {
  awk -v value="$1" 'BEGIN { exit !(value == 0) }'
}

require_idle_and_cold() {
  PRINT_JSON=$(moonraker_get 'printer/objects/query?print_stats') || {
    echo "ERROR: Cannot read print state from Moonraker"
    return 1
  }
  case "$PRINT_JSON" in
    *'"state": "standby"'*|*'"state":"standby"'*) ;;
    *)
      echo "ERROR: Printer is not in standby; refusing calibration motion"
      return 1
      ;;
  esac

  for HEATER in extruder heater_bed; do
    HEATER_JSON=$(moonraker_get "printer/objects/query?${HEATER}") || {
      echo "ERROR: Cannot read ${HEATER} state from Moonraker"
      return 1
    }
    TARGET=$(printf '%s\n' "$HEATER_JSON" | sed -n 's/.*"target": \([-0-9.]*\).*/\1/p')
    if [ -z "$TARGET" ] || ! numeric_zero "$TARGET"; then
      echo "ERROR: ${HEATER} target is ${TARGET:-unknown}; refusing calibration motion"
      return 1
    fi
  done
}

# Helper: wait for a CSV file to appear and stabilize (stop growing)
wait_for_csv() {
  TARGET="$1"
  TIMEOUT=300
  COUNT=0
  PREV_SIZE=0
  STABLE=0

  while [ $COUNT -lt $TIMEOUT ]; do
    if [ -f "$TARGET" ]; then
      CUR_SIZE=$(wc -c < "$TARGET" 2>/dev/null || echo 0)
      if [ "$CUR_SIZE" -gt 1000000 ] && [ "$CUR_SIZE" -eq "$PREV_SIZE" ]; then
        STABLE=$((STABLE + 1))
        if [ $STABLE -ge 2 ]; then
          echo "$CUR_SIZE"
          return 0
        fi
      else
        STABLE=0
      fi
      PREV_SIZE=$CUR_SIZE
    fi
    sleep 3
    COUNT=$((COUNT + 3))
  done
  echo "0"
  return 1
}

if [ ! -r "$TOKEN_FILE" ]; then
  echo "ERROR: ShakeTune token is missing or unreadable: $TOKEN_FILE"
  exit 1
fi
TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
if [ -z "$TOKEN" ]; then
  echo "ERROR: ShakeTune token is empty: $TOKEN_FILE"
  exit 1
fi

# Clean up old files
rm -f "$FILE_A" "$FILE_B"
require_idle_and_cold

echo "=== Belt A (1,-1) ==="
echo "Homing..."
send_gcode "G28"
sleep 20
echo "Centering toolhead..."
send_gcode "G0 X110 Y110 Z50 F6000"
sleep 3
echo "Running Belt A resonance test..."
send_gcode "TEST_RESONANCES AXIS=1,-1 OUTPUT=raw_data NAME=a FREQ_START=${FREQ_START} FREQ_END=${FREQ_END} HZ_PER_SEC=${HZ_PER_SEC}"

echo "Waiting for Belt A CSV..."
if ! SIZE_A=$(wait_for_csv "$FILE_A"); then
  echo "ERROR: Belt A CSV not ready"
  exit 1
fi
echo "Belt A complete: ${SIZE_A} bytes"

echo ""
echo "=== Firmware Restart ==="
firmware_restart
require_idle_and_cold
echo "Klipper ready"

echo ""
echo "=== Belt B (1,1) ==="
echo "Homing..."
send_gcode "G28"
sleep 20
echo "Centering toolhead..."
send_gcode "G0 X110 Y110 Z50 F6000"
sleep 3
echo "Running Belt B resonance test..."
send_gcode "TEST_RESONANCES AXIS=1,1 OUTPUT=raw_data NAME=b FREQ_START=${FREQ_START} FREQ_END=${FREQ_END} HZ_PER_SEC=${HZ_PER_SEC}"

echo "Waiting for Belt B CSV..."
if ! SIZE_B=$(wait_for_csv "$FILE_B"); then
  echo "ERROR: Belt B CSV not ready"
  exit 1
fi
echo "Belt B complete: ${SIZE_B} bytes"

echo ""
echo "=== Final Firmware Restart ==="
firmware_restart
require_idle_and_cold
echo "Klipper ready"

echo ""
echo "=== Upload ==="
BASE_URL="$BASE_URL" sh "${SCRIPT_DIR}/upload_belts.sh" "$HOST" "$PORT" "$PRINTER"
echo "Done!"
