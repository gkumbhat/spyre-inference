#!/usr/bin/env bash
# Time a server's boot the way run_vllm_benchmarks.py does: launch, poll /health, stop.
# Usage: boot_time.sh SERVER_CMD_FILE LOG_FILE
set -u
cmd_file=$1 log=$2
start=$SECONDS
setsid bash -c "$(cat "$cmd_file")" > "$log" 2>&1 &
pgid=$!
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
    if ! kill -0 "$pgid" 2>/dev/null; then echo "server exited before ready"; exit 1; fi
    sleep 2
done
echo "Server ready after $((SECONDS - start))s"
kill -TERM -- "-$pgid"
wait "$pgid" 2>/dev/null
