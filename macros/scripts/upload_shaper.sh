#!/bin/sh
# Upload shaper CSV files to remote Shake&Tune service
# Usage: upload_shaper.sh [HOST] [PORT] [PRINTER]

set -eu

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
UPLOAD_URL="${BASE_URL%/}"
TOKEN_FILE="${SHAKETUNE_TOKEN_FILE:-/usr/data/printer_data/config/shaketune/token}"

FILE_X="/tmp/raw_data_x_x.csv"
FILE_Y="/tmp/raw_data_y_y.csv"

if [ ! -r "$TOKEN_FILE" ]; then
  echo "ERROR: ShakeTune token is missing or unreadable: $TOKEN_FILE"
  exit 1
fi
TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
if [ -z "$TOKEN" ]; then
  echo "ERROR: ShakeTune token is empty: $TOKEN_FILE"
  exit 1
fi

echo "Waiting for CSV files to be written..."
TIMEOUT=30
COUNT=0
while [ $COUNT -lt $TIMEOUT ]; do
  SIZE_X=$(wc -c < "$FILE_X" 2>/dev/null || echo 0)
  SIZE_Y=$(wc -c < "$FILE_Y" 2>/dev/null || echo 0)
  if [ "$SIZE_X" -gt 1000000 ] && [ "$SIZE_Y" -gt 1000000 ]; then
    break
  fi
  sleep 1
  COUNT=$((COUNT + 1))
done

if [ "$SIZE_X" -gt 1000000 ] && [ "$SIZE_Y" -gt 1000000 ]; then
  TS=$(date +%Y%m%d_%H%M%S)

  # Compress files for faster upload (K1's BusyBox curl struggles with large files)
  echo "Compressing files..."
  gzip -f -k "$FILE_X"
  gzip -f -k "$FILE_Y"

  echo "Uploading to service..."
  RESPONSE=$(curl --timeout 360 -H "X-ShakeTune-Token: ${TOKEN}" POST "${UPLOAD_URL}/shaper" \
    -F "file_x=@${FILE_X}.gz" \
    -F "file_y=@${FILE_Y}.gz" \
    -F "printer=${PRINTER}" \
    -F "timestamp=${TS}") || {
      STATUS=$?
      rm -f "${FILE_X}.gz" "${FILE_Y}.gz"
      echo "ERROR: Upload failed with status ${STATUS}"
      exit "$STATUS"
  }
  COMPACT_RESPONSE="$(printf '%s' "$RESPONSE" | tr -d '[:space:]')"
  for AXIS in x y; do
    case "$COMPACT_RESPONSE" in
      *'"axis":"'"${AXIS}"'"'*) ;;
      *)
        echo "ERROR: Service did not confirm the ${AXIS}-axis graph"
        echo "$RESPONSE"
        exit 1
        ;;
    esac
  done

  for AXIS in x y; do
    RESULT_URL="${BASE_URL}/results/${PRINTER}/${TS}_shaper_${AXIS}.png"
    if ! wget -q -O /dev/null -T 30 "$RESULT_URL"; then
      echo "ERROR: Generated graph is not retrievable: $RESULT_URL"
      exit 1
    fi
  done

  # Cleanup compressed files
  rm -f "${FILE_X}.gz" "${FILE_Y}.gz"

  echo ""
  echo "==========================================="
  echo "X GRAPH: ${BASE_URL}/results/${PRINTER}/${TS}_shaper_x.png"
  echo "Y GRAPH: ${BASE_URL}/results/${PRINTER}/${TS}_shaper_y.png"
  echo "==========================================="
else
  echo "ERROR: Files not ready after ${TIMEOUT}s"
  echo "X: ${SIZE_X} bytes, Y: ${SIZE_Y} bytes"
  exit 1
fi
