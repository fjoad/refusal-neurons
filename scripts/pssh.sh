#!/bin/bash
# Runs a command on panther, retrying when the connection drops (ssh exit 255).
# Bypasses the shared ControlMaster connection, which goes stale after a VPN drop.
# Usage: scripts/pssh.sh 'cd ~/refusal-neurons && git pull -q'
for i in 1 2 3 4 5 6 7 8; do
  out=$(ssh -o ControlPath=none -o ConnectTimeout=20 -o ServerAliveInterval=10 panther.qcri.org "$@" 2>&1); rc=$?
  if [ $rc -ne 255 ]; then printf '%s\n' "$out"; exit $rc; fi
  sleep 6
done
echo "pssh: gave up after 8 connection drops" >&2
exit 255
