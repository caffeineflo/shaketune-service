#!/bin/sh
# Restart Klipper firmware and prove the restart cleared the homed-axis state.

set -eu

MOONRAKER_URL="${MOONRAKER_URL:-http://localhost:7125}"
RESTART_TIMEOUT_SECONDS="${SHAKETUNE_RESTART_TIMEOUT_SECONDS:-90}"
POLL_INTERVAL_SECONDS="${SHAKETUNE_RESTART_POLL_SECONDS:-2}"

moonraker_get() {
  wget -q -O - -T 10 "${MOONRAKER_URL%/}/$1" 2>/dev/null
}

toolhead_state="$(moonraker_get 'printer/objects/query?toolhead')" || {
  echo "ERROR: Cannot read toolhead state before firmware restart"
  exit 1
}
case "$toolhead_state" in
  *'"homed_axes": "xyz"'*|*'"homed_axes":"xyz"'*) ;;
  *)
    echo "ERROR: Toolhead was not fully homed before firmware restart"
    exit 1
    ;;
esac

# The K1 curl process may lose its connection while firmware restarts, so the
# observable homed-axis transition below is the authoritative success check.
curl --timeout 10 POST "${MOONRAKER_URL%/}/printer/firmware_restart" >/dev/null 2>&1 || true
sleep 5

elapsed=0
while [ "$elapsed" -lt "$RESTART_TIMEOUT_SECONDS" ]; do
  info="$(moonraker_get 'printer/info' || true)"
  toolhead_state="$(moonraker_get 'printer/objects/query?toolhead' || true)"
  case "$info:$toolhead_state" in
    *'"state": "ready"'*':'*'"homed_axes": ""'*|*'"state":"ready"'*':'*'"homed_axes":""'*)
      echo "Firmware restart verified: Klipper is ready and homed_axes is clear"
      exit 0
      ;;
  esac
  sleep "$POLL_INTERVAL_SECONDS"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done

echo "ERROR: Firmware restart was not verified within ${RESTART_TIMEOUT_SECONDS} seconds"
exit 1
