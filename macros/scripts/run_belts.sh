#!/bin/sh
# Run belt comparison test with firmware restart between sweeps to avoid
# "Move queue overflow" on K1's 209MB RAM.
#
# This script orchestrates the full sequence via Moonraker API:
#   1. Home + center + run Belt A
#   2. Firmware restart (clears move queue)
#   3. Wait for Klipper ready
#   4. Home + center + run Belt B
#   5. Upload both CSVs to shaketune service
#
# Must be launched detached from Klipper (via setsid) because the
# firmware restart would kill a Klipper-owned child process.
#
# Usage: run_belts.sh HOST PORT PRINTER [FREQ_START] [FREQ_END] [HZ_PER_SEC]

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
FREQ_START="${4:-5}"
FREQ_END="${5:-133.33}"
HZ_PER_SEC="${6:-1}"

FILE_A="/tmp/raw_data_axis=1.000,-1.000_a.csv"
FILE_B="/tmp/raw_data_axis=1.000,1.000_b.csv"

# Helper: send gcode via Moonraker
send_gcode() {
  ENCODED=$(echo "$1" | sed 's/ /%20/g; s/=/%3D/g; s/,/%2C/g')
  wget -q -O /dev/null "http://localhost:7125/printer/gcode/script?script=${ENCODED}" 2>/dev/null
}

# Helper: wait for Klipper to reach "ready" state
wait_ready() {
  TIMEOUT=60
  COUNT=0
  while [ $COUNT -lt $TIMEOUT ]; do
    STATE=$(wget -q -O - "http://localhost:7125/printer/info" 2>/dev/null)
    case "$STATE" in
      *'"state": "ready"'*|*'"state":"ready"'*)
        return 0
        ;;
    esac
    sleep 2
    COUNT=$((COUNT + 2))
  done
  echo "ERROR: Klipper not ready after ${TIMEOUT}s"
  return 1
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

# Clean up old files
rm -f "$FILE_A" "$FILE_B"

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
SIZE_A=$(wait_for_csv "$FILE_A")
if [ "$SIZE_A" = "0" ]; then
  echo "ERROR: Belt A CSV not ready"
  exit 1
fi
echo "Belt A complete: ${SIZE_A} bytes"

echo ""
echo "=== Firmware Restart ==="
wget -q -O /dev/null "http://localhost:7125/printer/firmware_restart" 2>/dev/null
sleep 5
echo "Waiting for Klipper..."
wait_ready || exit 1
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
SIZE_B=$(wait_for_csv "$FILE_B")
if [ "$SIZE_B" = "0" ]; then
  echo "ERROR: Belt B CSV not ready"
  exit 1
fi
echo "Belt B complete: ${SIZE_B} bytes"

echo ""
echo "=== Upload ==="
TS=$(date +%Y%m%d_%H%M%S)

# Compress files for faster upload
echo "Compressing files..."
gzip -f -k "$FILE_A"
gzip -f -k "$FILE_B"

echo "Uploading to service..."
curl POST "http://${HOST}:${PORT}/belts" \
  -F "file_a=@${FILE_A}.gz" \
  -F "file_b=@${FILE_B}.gz" \
  -F "printer=${PRINTER}" \
  -F "timestamp=${TS}" 2>/dev/null

# Cleanup
rm -f "${FILE_A}.gz" "${FILE_B}.gz"

echo ""
echo "==========================================="
echo "BELTS GRAPH: ${BASE_URL}/results/${PRINTER}/${TS}_belts.png"
echo "==========================================="
echo "Done!"
