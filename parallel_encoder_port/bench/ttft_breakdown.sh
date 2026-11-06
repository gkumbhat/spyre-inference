#!/usr/bin/env bash
# Split TTFT into queue and prefill time with vLLM's own per-request histograms.
# Starts the arm's recorded server command, runs a short copy of its bench command, then
# scrapes /metrics before stopping the server. Usage:
#   ttft_breakdown.sh SERVER_CMD_FILE BENCH_CMD_FILE NUM_PROMPTS OUT_DIR
set -u
server_cmd=$1 bench_cmd=$2 n=$3 out=$4
mkdir -p "$out"
setsid bash -c "$(cat "$server_cmd")" > "$out/server.log" 2>&1 &
pgid=$!
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
    kill -0 "$pgid" 2>/dev/null || { echo "server exited before ready"; exit 1; }
    sleep 5
done
bench=$(sed -E "s/--num-prompts [0-9]+/--num-prompts $n/; s#--result-dir [^ ]+#--result-dir $out#" "$bench_cmd")
bash -c "$bench" > "$out/bench.log" 2>&1
curl -s http://127.0.0.1:8000/metrics > "$out/metrics.txt"
kill -TERM -- "-$pgid"
wait "$pgid" 2>/dev/null
echo "done: $out"
