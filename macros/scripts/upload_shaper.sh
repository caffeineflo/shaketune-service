#!/bin/sh
# Upload shaper CSV files to remote Shake&Tune service
# Usage: upload_shaper.sh [HOST] [PORT] [PRINTER]

HOST="${1:-192.168.1.100}"
PORT="${2:-8080}"
PRINTER="${3:-default}"

FILE_X="/tmp/raw_data_x_x.csv"
FILE_Y="/tmp/raw_data_y_y.csv"

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
  echo "Uploading to service..."
  curl -X POST "http://${HOST}:${PORT}/shaper" \
    -F "file_x=@${FILE_X}" \
    -F "file_y=@${FILE_Y}" \
    -F "printer=${PRINTER}" \
    -F "timestamp=${TS}" 2>/dev/null
  echo ""
  echo "==========================================="
  echo "SHAPER GRAPH: http://${HOST}:${PORT}/results/${PRINTER}/${TS}_shaper.png"
  echo "==========================================="
else
  echo "ERROR: Files not ready after ${TIMEOUT}s"
  echo "X: ${SIZE_X} bytes, Y: ${SIZE_Y} bytes"
  exit 1
fi
