#!/usr/bin/env bash
# Run one Isaac workload with a two-slot GPU gate and lane-scoped cleanup.
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MAX_SECONDS LOG_PATH COMMAND [ARG ...]" >&2
  exit 64
fi
if [[ ! ${CPGEN_LANE_ID:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-7]$ ]]; then
  echo "CPGEN_LANE_ID and one numeric CUDA_VISIBLE_DEVICES slot are required" >&2
  exit 64
fi

max_seconds=$1
log_path=$2
shift 2
stall_seconds=${ISAAC_STALL_SECONDS:-120}
if [[ ! $max_seconds =~ ^[1-9][0-9]*$ || ! $stall_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "guard and stall durations must be positive integers" >&2
  exit 64
fi

mkdir -p "$(dirname "$log_path")"

# A lane may own only one guarded workload.  Keep this lock for the complete
# run so simultaneous same-lane launchers cannot both pass a zero-worker
# snapshot before either child becomes visible.
exec 6>"/tmp/cpgen-isaac-lane-${CPGEN_LANE_ID}.lock"
if ! flock -n 6; then
  echo "LANE_WORKER_LOCK=FAIL lane=${CPGEN_LANE_ID}" | tee "${log_path}.gate"
  exit 41
fi

# Serialize only admission, then hold one of two slot locks for the run.  This
# permits two independent lanes on one physical GPU without an exclusive
# GPU-wide run lock.  If one pre-slot worker is already visible, reserve its
# capacity by holding both free slots until this run exits.
exec 7>"/tmp/cpgen-isaac-gpu-${CUDA_VISIBLE_DEVICES}.admission.lock"
flock 7

gpu_pids=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | awk 'NF && $1 != "[N/A]" {print $1}' | sort -un || true)
gpu_occupancy=$(printf '%s\n' "$gpu_pids" | awk 'NF {count++} END {print count + 0}')
minimum_free_mib=${CPGEN_MIN_FREE_GPU_MIB:-12000}
gpu_free_mib=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk 'NF {print int($1); exit}' || true)
if [[ $gpu_occupancy -ge 2 || -z $gpu_free_mib || $gpu_free_mib -lt $minimum_free_mib ]]; then
  {
    echo "GPU_CAPACITY_GATE=FAIL gpu=${CUDA_VISIBLE_DEVICES} occupancy=$gpu_occupancy limit=2 free_mib=${gpu_free_mib:-unknown} required_free_mib=$minimum_free_mib"
    echo "$gpu_pids"
  } | tee "${log_path}.gate"
  exit 41
fi

exec 9>"/tmp/cpgen-isaac-gpu-${CUDA_VISIBLE_DEVICES}.slot0.lock"
exec 8>"/tmp/cpgen-isaac-gpu-${CUDA_VISIBLE_DEVICES}.slot1.lock"
slot0=0
slot1=0
flock -n 9 && slot0=1
flock -n 8 && slot1=1
if [[ $gpu_occupancy -eq 0 && $slot0 -eq 1 ]]; then
  slot=0
  [[ $slot1 -eq 1 ]] && flock -u 8
elif [[ $gpu_occupancy -eq 0 && $slot1 -eq 1 ]]; then
  slot=1
elif [[ $gpu_occupancy -eq 1 && $slot0 -eq 1 && $slot1 -eq 1 ]]; then
  slot="0,1-reserving-existing"
elif [[ $gpu_occupancy -eq 1 && $slot0 -eq 1 ]]; then
  slot=0
elif [[ $gpu_occupancy -eq 1 && $slot1 -eq 1 ]]; then
  slot=1
else
  echo "GPU_CAPACITY_SLOT=FAIL gpu=${CUDA_VISIBLE_DEVICES} occupancy=$gpu_occupancy limit=2" | tee "${log_path}.gate"
  exit 41
fi
flock -u 7
exec 7>&-

lane_worker_snapshot() {
  local pid env_file comm args
  # Inspect only possible simulator/runner processes.  Walking every
  # /proc/*/environ is both racy and slow enough on shared nodes to exceed the
  # guard's own outer timeout after a workload has already exited.
  while read -r pid; do
    [[ -n $pid ]] || continue
    env_file="/proc/$pid/environ"
    if ! tr '\0' '\n' <"$env_file" 2>/dev/null \
      | grep -Fxq "CPGEN_LANE_ID=${CPGEN_LANE_ID}"; then
      continue
    fi
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    args=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ $comm == isaac-sim || $comm == kit || ( $comm == python* && $args == *run_hangmug_skill_program.py* ) ]]; then
      printf '%s %s %s\n' "$pid" "$comm" "$args"
    fi
  done < <(
    {
      pgrep -x isaac-sim 2>/dev/null || true
      pgrep -x kit 2>/dev/null || true
      pgrep -f '[r]un_hangmug_skill_program.py' 2>/dev/null || true
    } | sort -un
  )
}

# `kill -0` also succeeds for an exited child that has become a zombie.  The
# guard is that child's parent, so waiting for `kill -0` to fail before calling
# `wait` deadlocks forever and retains the GPU flock.  Treat terminal proc
# states as exited, then let the existing `wait` below reap the child and
# preserve its real return code.
child_is_running() {
  local pid=$1 state
  [[ -r /proc/$pid/status ]] || return 1
  state=$(awk '$1 == "State:" { print $2; exit }' "/proc/$pid/status" 2>/dev/null || true)
  [[ -n $state && $state != Z && $state != X ]]
}

{
  date --iso-8601=seconds
  echo "ZERO_WORKER_GATE=PASS"
  echo "LANE_WORKER_LOCK=PASS lane=${CPGEN_LANE_ID}"
  echo "GPU_CAPACITY_GATE=PASS gpu=${CUDA_VISIBLE_DEVICES} occupancy_before=$gpu_occupancy limit=2 free_mib=$gpu_free_mib required_free_mib=$minimum_free_mib slot=$slot lane=${CPGEN_LANE_ID}"
  echo "STEADY_STATE_GUARD_SECONDS=$max_seconds"
  echo "NO_STEP_PROGRESS_STALL_SECONDS=$stall_seconds"
  printf 'COMMAND='
  printf '%q ' "$@"
  printf '\n'
} | tee "${log_path}.gate"

# Do not combine the background workload with process substitution here.  On
# some Bash builds `$!` can identify the `tee` process-substitution shell
# instead of `timeout`, creating a circular wait after the real workload exits.
# Write directly to the durable lane log so `run_pid` always identifies the
# timeout process whose exit status we must preserve.
timeout --signal=TERM --kill-after=30s "${max_seconds}s" "$@" >"$log_path" 2>&1 &
run_pid=$!
last_progress=""
last_progress_time=$(date +%s)
progress_started=0
stall_triggered=0
while child_is_running "$run_pid"; do
  progress=$(grep -E 'STEP_PROGRESS([^0-9]|$)' "$log_path" 2>/dev/null | tail -n 1 || true)
  now=$(date +%s)
  if [[ -n $progress && $progress != "$last_progress" ]]; then
    last_progress=$progress
    last_progress_time=$now
    progress_started=1
  elif [[ $progress_started -eq 1 && $((now - last_progress_time)) -ge $stall_seconds ]]; then
    stall_triggered=1
    echo "NO_STEP_PROGRESS_TIMEOUT seconds=$stall_seconds last=$last_progress" | tee -a "$log_path"
    kill -TERM "$run_pid" 2>/dev/null || true
    break
  fi
  sleep 5
done

if [[ $stall_triggered -eq 1 ]]; then
  for _ in 1 2 3 4 5 6; do
    child_is_running "$run_pid" || break
    sleep 5
  done
  kill -KILL "$run_pid" 2>/dev/null || true
fi
wait "$run_pid"
run_rc=$?
workload_failure_marker=$(grep -E '^[A-Z][A-Z0-9_]*_FAILED([[:space:]]|$)' "$log_path" 2>/dev/null | tail -n 1 || true)
if [[ -n $workload_failure_marker && $run_rc -eq 0 ]]; then
  run_rc=1
fi
echo "NO_STEP_PROGRESS_STALL_TRIGGERED=$stall_triggered" | tee "${log_path}.stall"

remaining=$(lane_worker_snapshot)
if [[ -n $remaining ]]; then
  {
    echo "POST_RUN_ZERO_WORKER=FAIL"
    echo "$remaining"
  } | tee -a "$log_path"
  while read -r pid _; do
    [[ -n $pid ]] && kill -TERM "$pid" 2>/dev/null || true
  done <<<"$remaining"
  sleep 5
  remaining=$(lane_worker_snapshot)
  if [[ -n $remaining ]]; then
    while read -r pid _; do
      [[ -n $pid ]] && kill -KILL "$pid" 2>/dev/null || true
    done <<<"$remaining"
    echo "POST_RUN_FORCED_KILL=1" | tee -a "$log_path"
  fi
else
  echo "POST_RUN_ZERO_WORKER=PASS" | tee -a "$log_path"
fi

{
  [[ -n $workload_failure_marker ]] && echo "WORKLOAD_FAILURE_MARKER=$workload_failure_marker"
  echo "GUARDED_RUN_EXIT=$run_rc"
} | tee "${log_path}.exit"
exit "$run_rc"
