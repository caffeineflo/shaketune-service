#!/bin/sh
# Upload belts CSV files to remote Shake&Tune service
# Usage: upload_belts.sh [HOST] [PORT] [PRINTER]

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"

FILE_A="/tmp/raw_data_axis=1.000,-1.000_a.csv"
FILE_B="/tmp/raw_data_axis=1.000,1.000_b.csv"

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
  echo "Uploading to service..."
  curl POST "http://${HOST}:${PORT}/belts" \
    -F "file_a=@${FILE_A}" \
    -F "file_b=@${FILE_B}" \
    -F "printer=${PRINTER}" \
    -F "timestamp=${TS}" 2>/dev/null
  echo ""
  echo "==========================================="
  echo "BELTS GRAPH: http://${HOST}:${PORT}/results/${PRINTER}/${TS}_belts.png"
  echo "==========================================="
else
  echo "ERROR: Files not ready after ${TIMEOUT}s"
  echo "A: ${SIZE_A} bytes, B: ${SIZE_B} bytes"
  exit 1
fi
