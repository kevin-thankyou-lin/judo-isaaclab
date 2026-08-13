#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GUARD" >&2
  exit 64
fi

guard=$(realpath "$1")
test_root=$(mktemp -d /tmp/cpgen-two-worker-guard-test.XXXXXX)
trap 'rm -rf -- "$test_root"' EXIT
mkdir -p "$test_root/bin"

cat >"$test_root/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ $* == *query-compute-apps=pid* ]]; then
  printf '%s\n' "${FAKE_GPU_PIDS:-}"
  exit 0
fi
if [[ $* == *query-gpu=memory.free* ]]; then
  echo 48000
  exit 0
fi
exit 2
EOF
chmod +x "$test_root/bin/nvidia-smi"

run_guard() {
  local lane=$1 log=$2
  CPGEN_LANE_ID="$lane" CUDA_VISIBLE_DEVICES=7 \
    FAKE_GPU_PIDS="${FAKE_GPU_PIDS:-}" \
    PATH="$test_root/bin:$PATH" ISAAC_STALL_SECONDS=30 \
    "$guard" 20 "$log" bash -c \
      'printf "%s\n" "$CPGEN_LANE_ID" >>"$1"; sleep 2' _ \
      "$test_root/started"
}

set +e
run_guard guard-regression-a "$test_root/a.log" & p1=$!
run_guard guard-regression-b "$test_root/b.log" & p2=$!
run_guard guard-regression-c "$test_root/c.log" & p3=$!
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
set -e

mapfile -t statuses < <(printf '%s\n' "$r1" "$r2" "$r3" | sort -n)
[[ ${statuses[*]} == "0 0 41" ]]
[[ $(wc -l <"$test_root/started") -eq 2 ]]
[[ $(grep -l 'GPU_CAPACITY_GATE=PASS.*limit=2' "$test_root"/{a,b,c}.log.gate | wc -l) -eq 2 ]]
[[ $(grep -l 'GPU_CAPACITY_SLOT=FAIL' "$test_root"/{a,b,c}.log.gate | wc -l) -eq 1 ]]

set +e
run_guard guard-regression-same "$test_root/same-a.log" & same1=$!
run_guard guard-regression-same "$test_root/same-b.log" & same2=$!
wait "$same1"; same_r1=$?
wait "$same2"; same_r2=$?
set -e
mapfile -t same_statuses < <(printf '%s\n' "$same_r1" "$same_r2" | sort -n)
[[ ${same_statuses[*]} == "0 41" ]]
[[ $(grep -l 'LANE_WORKER_LOCK=FAIL' "$test_root"/same-{a,b}.log.gate | wc -l) -eq 1 ]]

FAKE_GPU_PIDS=101 run_guard guard-regression-one-existing "$test_root/one.log"
grep -q 'GPU_CAPACITY_GATE=PASS.*occupancy_before=1.*limit=2' "$test_root/one.log.gate"
set +e
FAKE_GPU_PIDS=$'101\n102' run_guard guard-regression-two-existing "$test_root/two.log"
two_rc=$?
set -e
[[ $two_rc -eq 41 ]]
grep -q 'GPU_CAPACITY_GATE=FAIL.*occupancy=2.*limit=2' "$test_root/two.log.gate"

set +e
CPGEN_LANE_ID=guard-regression-exit7 CUDA_VISIBLE_DEVICES=7 \
  PATH="$test_root/bin:$PATH" ISAAC_STALL_SECONDS=30 \
  "$guard" 10 "$test_root/exit7.log" bash -c 'exit 7'
exit7_rc=$?
set -e
[[ $exit7_rc -eq 7 ]]
grep -Fxq 'GUARDED_RUN_EXIT=7' "$test_root/exit7.log.exit"

echo "ATOMIC_MAX2_REGRESSION=PASS admitted=2 rejected=1"
echo "SAME_LANE_SINGLE_WORKER_REGRESSION=PASS admitted=1 rejected=1"
echo "OCCUPANCY_BOUNDARY_REGRESSION=PASS occupancy1=admit occupancy2=reject"
echo "EXIT7_PRESERVATION=PASS"
