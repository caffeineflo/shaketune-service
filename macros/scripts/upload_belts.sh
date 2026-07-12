#!/bin/sh
# Upload belts CSV files to remote Shake&Tune service
# Usage: upload_belts.sh [HOST] [PORT] [PRINTER]

set -eu

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
UPLOAD_URL="${BASE_URL%/}"
TOKEN_FILE="${SHAKETUNE_TOKEN_FILE:-/usr/data/printer_data/config/shaketune/token}"

FILE_A="${SHAKETUNE_BELT_FILE_A:-/tmp/raw_data_axis=1.000,-1.000_a.csv}"
FILE_B="${SHAKETUNE_BELT_FILE_B:-/tmp/raw_data_axis=1.000,1.000_b.csv}"

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
  SIZE_A=$(wc -c < "$FILE_A" 2>/dev/null || echo 0)
  SIZE_B=$(wc -c < "$FILE_B" 2>/dev/null || echo 0)
  echo "A=${SIZE_A} bytes, B=${SIZE_B} bytes"
  if [ "$SIZE_A" -gt 1000000 ] && [ "$SIZE_B" -gt 1000000 ]; then
    echo "Both files ready"
    break
  fi
  sleep 1
  COUNT=$((COUNT + 1))
done

if [ "$SIZE_A" -gt 1000000 ] && [ "$SIZE_B" -gt 1000000 ]; then
  TS=$(date +%Y%m%d_%H%M%S)

  # Compress files for faster upload (K1's BusyBox curl struggles with large files)
  echo "Compressing files..."
  gzip -f -k "$FILE_A"
  gzip -f -k "$FILE_B"

  echo "Uploading to service..."
  RESPONSE=$(curl --timeout 360 -H "X-ShakeTune-Token: ${TOKEN}" POST "${UPLOAD_URL}/belts" \
    -F "file_a=@${FILE_A}.gz" \
    -F "file_b=@${FILE_B}.gz" \
    -F "printer=${PRINTER}" \
    -F "timestamp=${TS}") || {
      STATUS=$?
      echo "ERROR: Upload failed with status ${STATUS}"
      exit "$STATUS"
    }

  COMPACT_RESPONSE="$(printf '%s' "$RESPONSE" | tr -d '[:space:]')"
  case "$COMPACT_RESPONSE" in
    *'"url":"/results/'"${PRINTER}"'/'"${TS}"'_belts.png"'*) ;;
    *)
      echo "ERROR: Service did not confirm the belts graph"
      echo "$RESPONSE"
      exit 1
      ;;
  esac

  RESULT_URL="${BASE_URL}/results/${PRINTER}/${TS}_belts.png"
  if ! wget -q -O /dev/null -T 30 "$RESULT_URL"; then
    echo "ERROR: Generated graph is not retrievable: $RESULT_URL"
    exit 1
  fi

  # Cleanup only after the service response and graph retrieval are verified.
  rm -f "$FILE_A" "$FILE_B" "${FILE_A}.gz" "${FILE_B}.gz"

  echo ""
  echo "==========================================="
  echo "BELTS GRAPH: ${BASE_URL}/results/${PRINTER}/${TS}_belts.png"
  echo "==========================================="
else
  echo "ERROR: Files not ready after ${TIMEOUT}s"
  echo "A: ${SIZE_A} bytes, B: ${SIZE_B} bytes"
  exit 1
fi
