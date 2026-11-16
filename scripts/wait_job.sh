#!/bin/bash
# Returns when any of the given panther SLURM jobs has left the queue (finished, failed or cancelled).
# Polls every 5 minutes with short ssh calls; a failed call counts as "unknown", not "finished".
# Usage (run in the background): scripts/wait_job.sh <jobid> [<jobid> ...]
dir=$(cd "$(dirname "$0")" && pwd)
while true; do
  q=$("$dir/pssh.sh" 'source /etc/profile.d/modules.sh; module load slurm; squeue -u $USER -h -o "%i %T" && echo QUEUE_OK')
  if [[ "$q" == *QUEUE_OK* ]]; then
    for j in "$@"; do
      if ! grep -q "^$j " <<< "$q"; then
        echo "left the queue: $j"
        echo "$q" | grep -v QUEUE_OK | tr '\n' ';'
        exit 0
      fi
    done
  fi
  sleep 300
done
