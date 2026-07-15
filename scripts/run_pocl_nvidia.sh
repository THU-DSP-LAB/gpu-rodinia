#!/usr/bin/env bash

set -euo pipefail

readonly DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd "$DIR/../.." && pwd)"
readonly OCLDIR="$DIR/../opencl"
readonly DEFAULT_BENCHMARKS="b+tree backprop bfs cfd dwt2d gaussian heartwall hotspot hotspot3D hybridsort kmeans lavaMD leukocyte lud myocyte nn nw particlefilter pathfinder srad streamcluster"
readonly DEFAULT_OUTPUT_DIR="$DIR/results-pocl-nvidia"
readonly DEFAULT_POCL_INSTALL_DIR="$ROOT/pocl/build_cuda_stable/install"
readonly DEFAULT_NATIVE_PLATFORM=0
readonly DEFAULT_NATIVE_DEVICE=0
readonly DEFAULT_POCL_PLATFORM=0
readonly DEFAULT_POCL_DEVICE=0
readonly DEFAULT_POCL_ARCH="sm_89"
readonly DEFAULT_BENCHMARK_TIMEOUT_SECONDS=1800
readonly TIMEOUT_KILL_GRACE_SECONDS=5
readonly GPU_IDLE_WAIT_SECONDS=10
readonly GPU_IDLE_POLL_SECONDS=0.2

declare -A OUTPUT_FILES=(
  [b+tree]="output.txt"
  [backprop]="backprop_result.txt"
  [bfs]="bfs_result.txt"
  [cfd]="density.txt|momentum.txt|density_energy.txt"
  [dwt2d]="rgb.bmp.dwt.r|rgb.bmp.dwt.g|rgb.bmp.dwt.b"
  [gaussian]="gaussian_result.txt"
  [heartwall]="result.txt"
  [hotspot]="output.txt"
  [hotspot3D]="output.txt"
  [hybridsort]="hybridoutput.txt"
  [lavaMD]="result.txt"
  [leukocyte]="result.txt"
  [lud]="lud_verify.txt"
  [myocyte]="output.txt"
  [nn]="nn_result.txt"
  [nw]="nw_result.txt"
  [particlefilter]="output.txt"
  [pathfinder]="pathfinder_result.txt"
  [srad]="output/image_out.pgm"
  [streamcluster]="output.txt"
)

declare -A VALIDATE_RUN_ARGS=(
  [backprop]="--validate"
  [bfs]="--validate"
  [gaussian]="--validate"
  [hybridsort]="--validate"
  [lud]="-v"
  [nn]="--validate"
  [nw]="--traceback"
  [pathfinder]="--validate"
)

declare -A MAKE_ARGS=(
  [heartwall]="OUTPUT=1"
  [hybridsort]="VERIFY=1 OUTPUT=1"
  [lavaMD]="OUTPUT=1"
  [leukocyte]="OUTPUT=1"
)

declare -A RUNTIME_MODES=(
  [dwt2d]="context_device_index"
  [leukocyte]="autogpu"
)

declare -A POCL_DEVICE_OVERRIDES=(
  [dwt2d]=0
)

declare -A ABS_TOLERANCES=(
  [gaussian]="3e-5"
  [leukocyte]="5e-3"
  [myocyte]="1e-4"
)

declare -A REL_TOLERANCES=(
  [gaussian]="1e-4"
  [myocyte]="2e-2"
)

benchmarks="$DEFAULT_BENCHMARKS"
validate=0
iterations=1
native_platform=$DEFAULT_NATIVE_PLATFORM
native_device=$DEFAULT_NATIVE_DEVICE
pocl_platform=$DEFAULT_POCL_PLATFORM
pocl_device=$DEFAULT_POCL_DEVICE
pocl_arch=$DEFAULT_POCL_ARCH
output_dir=$DEFAULT_OUTPUT_DIR
pocl_install_dir=$DEFAULT_POCL_INSTALL_DIR
benchmark_timeout=$DEFAULT_BENCHMARK_TIMEOUT_SECONDS
require_cold_cache=0

usage() {
  cat <<'EOF'
Usage: run_pocl_nvidia.sh [options]

  --validate                 compare native NVIDIA OpenCL outputs with pocl GPU outputs
  --iterations N             run each backend N times (default: 1)
  --benchmarks "a b c"       space-separated benchmark list
  --native-platform N        native OpenCL platform id (default: 0)
  --native-device N          native OpenCL device id (default: 0)
  --pocl-platform N          pocl platform id (default: 0)
  --pocl-device N            pocl device id (default: 0)
  --pocl-arch ARCH           POCL_CUDA_GPU_ARCH value (default: sm_89)
  --pocl-install-dir DIR     target PoCL installation
  --output-dir DIR           evidence and isolated kernel-cache directory
  --cold-cache               fail if the isolated kernel cache already exists
  --timeout SECONDS          per backend run timeout; 0 disables (default: 1800)
EOF
}

while (($#)); do
  case "${1:-}" in
    --validate)
      validate=1
      shift
      ;;
    --iterations)
      iterations="${2:-}"
      shift 2
      ;;
    --benchmarks)
      benchmarks="${2:-}"
      shift 2
      ;;
    --native-platform)
      native_platform="${2:-}"
      shift 2
      ;;
    --native-device)
      native_device="${2:-}"
      shift 2
      ;;
    --pocl-platform)
      pocl_platform="${2:-}"
      shift 2
      ;;
    --pocl-device)
      pocl_device="${2:-}"
      shift 2
      ;;
    --pocl-arch)
      pocl_arch="${2:-}"
      shift 2
      ;;
    --pocl-install-dir)
      pocl_install_dir="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --cold-cache)
      require_cold_cache=1
      shift
      ;;
    --timeout)
      benchmark_timeout="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo "--iterations must be a positive integer" >&2
  exit 1
fi
if [[ ! "$benchmark_timeout" =~ ^[0-9]+$ ]]; then
  echo "--timeout must be a non-negative integer" >&2
  exit 1
fi

readonly OUTDIR="$(realpath -m "$output_dir")"
readonly POCL_INSTALL_DIR="$(realpath -m "$pocl_install_dir")"
readonly POCL_LIBDIR="$POCL_INSTALL_DIR/lib"
readonly POCL_LOADER_LINK="$POCL_LIBDIR/libOpenCL.so.1"
readonly POCL_CACHE_DIR="$OUTDIR/kernel-cache"

if ((require_cold_cache)) && [[ -e "$POCL_CACHE_DIR" ]]; then
  echo "Cold-cache run requested, but cache already exists: $POCL_CACHE_DIR" >&2
  exit 1
fi
mkdir -p "$OUTDIR"

ensure_pocl_loader() {
  if [[ ! -e "$POCL_LOADER_LINK" ]]; then
    echo "Missing PoCL loader: $POCL_LOADER_LINK" >&2
    return 1
  fi
  local resolved_loader
  resolved_loader="$(readlink -f "$POCL_LOADER_LINK")"
  if [[ "$resolved_loader" != "$POCL_LIBDIR/"* ]]; then
    echo "PoCL loader resolves outside target installation: $resolved_loader" >&2
    return 1
  fi
}

gpu_processes() {
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
}

wait_for_gpu_idle() {
  local deadline=$((SECONDS + GPU_IDLE_WAIT_SECONDS))
  local processes
  processes="$(gpu_processes)"
  while [[ -n "$processes" && $SECONDS -lt $deadline ]]; do
    sleep "$GPU_IDLE_POLL_SECONDS"
    processes="$(gpu_processes)"
  done
  [[ -z "$processes" ]] || printf '%s\n' "$processes"
}

assert_gpu_idle() {
  local stage=$1
  local processes
  if processes="$(wait_for_gpu_idle)"; then
    return 0
  fi
  echo "GPU is not idle $stage:" >&2
  printf '%s\n' "$processes" >&2
  return 1
}

verify_pocl_device() {
  local verifier="$ROOT/pocl/tools/scripts/cuda_stable/verify_device.py"
  env \
    "LD_LIBRARY_PATH=$POCL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    POCL_DEVICES=cuda \
    "$verifier" \
    --library "$POCL_LIBDIR/libOpenCL.so" \
    --output "$OUTDIR/device-contract.json"
}

record_environment() {
  {
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'benchmarks=%s\niterations=%s\n' "$benchmarks" "$iterations"
    printf 'pocl_install=%s\npocl_loader=%s\n' \
      "$POCL_INSTALL_DIR" "$(readlink -f "$POCL_LOADER_LINK")"
    printf 'pocl_cache=%s\npocl_arch=%s\ntimeout_seconds=%s\n' \
      "$POCL_CACHE_DIR" "$pocl_arch" "$benchmark_timeout"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid \
      --format=csv,noheader
    printf 'llvm=' && /usr/bin/llvm-config-18 --version
    printf 'pocl_commit=' && git -C "$ROOT/pocl" rev-parse HEAD
    printf 'rodinia_commit=' && git -C "$ROOT/gpu-rodinia" rev-parse HEAD
    git -C "$ROOT/pocl" status --short
    git -C "$ROOT/gpu-rodinia" status --short
  } > "$OUTDIR/environment.txt"
}

remove_outputs() {
  local benchmark=$1
  local outputs="${OUTPUT_FILES[$benchmark]-}"
  local out_file
  for out_file in ${outputs//|/ }; do
    rm -f "$OCLDIR/$benchmark/$out_file"
  done
}

copy_outputs() {
  local benchmark=$1
  local runtime=$2
  local iter=$3
  local outputs="${OUTPUT_FILES[$benchmark]-}"
  local idx=0
  local out_file
  for out_file in ${outputs//|/ }; do
    if [[ -f "$OCLDIR/$benchmark/$out_file" ]]; then
      cp "$OCLDIR/$benchmark/$out_file" \
        "$OUTDIR/$benchmark.$runtime.$iter.$idx.out"
    fi
    idx=$((idx + 1))
  done
}

build_benchmark() {
  local benchmark=$1
  local -a extra_args=()
  read -r -a extra_args <<< "${MAKE_ARGS[$benchmark]-}"
  (
    cd "$OCLDIR/$benchmark"
    make clean TYPE=GPU >/dev/null 2>&1 || true
    make -j"$(nproc)" TYPE=GPU "${extra_args[@]}"
  ) |& tee "$OUTDIR/$benchmark.make.log"
}

verify_benchmark_loader() {
  local benchmark=$1
  local expected_loader
  local found_opencl=0
  expected_loader="$(readlink -f "$POCL_LOADER_LINK")"

  while IFS= read -r -d '' executable; do
    local loader
    loader="$(
      LD_LIBRARY_PATH="$POCL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$executable" 2>/dev/null \
        | awk '$2 == "=>" && $3 ~ /^\// && $1 ~ /^libOpenCL[.]so/ { print $3; exit }'
    )"
    if [[ -z "$loader" ]]; then
      continue
    fi
    found_opencl=1
    loader="$(readlink -f "$loader")"
    if [[ "$loader" != "$expected_loader" ]]; then
      echo "$executable resolves OpenCL to $loader, expected $expected_loader" >&2
      return 1
    fi
  done < <(find "$OCLDIR/$benchmark" -maxdepth 1 -type f -perm /111 -print0)

  if ((found_opencl == 0)); then
    echo "No OpenCL-linked executable found for $benchmark" >&2
    return 1
  fi
}

run_benchmark() {
  local benchmark=$1
  local runtime=$2
  local iter=$3
  local log_file="$OUTDIR/$benchmark.$runtime.$iter.log"
  local mode="${RUNTIME_MODES[$benchmark]-standard}"
  local device_override="${POCL_DEVICE_OVERRIDES[$benchmark]-}"
  local -a run_cmd=(./run)
  local -a extra_args=()
  local -a envs=()
  local status=0

  remove_outputs "$benchmark"

  if ((validate)); then
    read -r -a extra_args <<< "${VALIDATE_RUN_ARGS[$benchmark]-}"
    run_cmd+=("${extra_args[@]}")
  fi

  case "$runtime" in
    native)
      if [[ "$mode" == "standard" ]]; then
        run_cmd+=(-p "$native_platform" -d "$native_device")
      elif [[ "$mode" == "context_device_index" ]]; then
        run_cmd+=(-p "$native_platform" -d "$native_device")
      fi
      ;;
    pocl)
      envs+=(
        "LD_LIBRARY_PATH=$POCL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        "POCL_CACHE_DIR=$POCL_CACHE_DIR"
        "POCL_CUDA_GPU_ARCH=$pocl_arch"
        "POCL_DEVICES=cuda"
      )
      if [[ "$mode" == "standard" ]]; then
        run_cmd+=(-p "$pocl_platform" -d "$pocl_device")
      elif [[ "$mode" == "context_device_index" ]]; then
        run_cmd+=(-p "$pocl_platform" -d "${device_override:-$pocl_device}")
      fi
      ;;
    *)
      echo "Unknown runtime: $runtime" >&2
      return 1
      ;;
  esac

  if ((benchmark_timeout == 0)); then
    (
      cd "$OCLDIR/$benchmark"
      env "${envs[@]}" "${run_cmd[@]}"
    ) |& tee "$log_file"
  else
    (
      cd "$OCLDIR/$benchmark"
      timeout --signal=TERM --kill-after="$TIMEOUT_KILL_GRACE_SECONDS" \
        "$benchmark_timeout" env "${envs[@]}" "${run_cmd[@]}"
    ) |& tee "$log_file"
  fi
  status=${PIPESTATUS[0]}

  copy_outputs "$benchmark" "$runtime" "$iter"
  if ! assert_gpu_idle "after $benchmark $runtime iteration $iter"; then
    return 1
  fi
  if ((status == 124)); then
    echo "  $benchmark $runtime iteration $iter timed out after ${benchmark_timeout}s" >&2
  fi
  return "$status"
}

log_has_runtime_error() {
  local log_file=$1
  grep -Eiq \
    '(^ERROR:|Assertion|Aborted|SIGFPE|SIGSEGV|Segmentation fault|core dumped|exception|CUDA_ERROR_|Failed to create|Could not create|CL_INVALID_|No devices available)' \
    "$log_file"
}

source "$DIR/rodinia_validation.sh"

ensure_pocl_loader
assert_gpu_idle "before Rodinia validation"
verify_pocl_device
record_environment

overall_pass=1
passed_benchmarks=0
failed_benchmarks=0
for benchmark in $benchmarks; do
  : > "$OUTDIR/$benchmark.txt"
  echo "$(date) # running $benchmark"

  if [[ ! -x "$OCLDIR/$benchmark/run" ]]; then
    chmod +x "$OCLDIR/$benchmark/run" 2>/dev/null || true
  fi

  if [[ ! -f "$OCLDIR/$benchmark/run" ]]; then
    echo "  missing run script for $benchmark" >&2
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
    overall_pass=0
    failed_benchmarks=$((failed_benchmarks + 1))
    continue
  fi

  if ! build_benchmark "$benchmark"; then
    echo "  build failed for $benchmark" >&2
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
    overall_pass=0
    failed_benchmarks=$((failed_benchmarks + 1))
    echo
    continue
  fi
  if ! verify_benchmark_loader "$benchmark"; then
    echo "  loader verification failed for $benchmark" >&2
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
    overall_pass=0
    failed_benchmarks=$((failed_benchmarks + 1))
    echo
    continue
  fi

  pass=1
  for iter in $(seq 1 "$iterations"); do
    if ! run_benchmark "$benchmark" native "$iter"; then
      pass=0
      echo "  native run failed for $benchmark (iter $iter)" >&2
    fi
    if ! run_benchmark "$benchmark" pocl "$iter"; then
      pass=0
      echo "  pocl run failed for $benchmark (iter $iter)" >&2
      continue
    fi
    if log_has_runtime_error "$OUTDIR/$benchmark.native.$iter.log"; then
      pass=0
      echo "  native log reported runtime error for $benchmark (iter $iter)" >&2
    fi
    if log_has_runtime_error "$OUTDIR/$benchmark.pocl.$iter.log"; then
      pass=0
      echo "  pocl log reported runtime error for $benchmark (iter $iter)" >&2
    fi
    if ((validate)) && ! validate_outputs "$benchmark" "$iter"; then
      pass=0
    fi
  done

  if ((pass)); then
    printf "validate=%d pass=%s\n\n" "$validate" "YES" >> "$OUTDIR/$benchmark.txt"
    passed_benchmarks=$((passed_benchmarks + 1))
  else
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
    overall_pass=0
    failed_benchmarks=$((failed_benchmarks + 1))
  fi

  echo
done

assert_gpu_idle "after Rodinia validation"
printf 'passed=%d\nfailed=%d\n' "$passed_benchmarks" "$failed_benchmarks" \
  | tee "$OUTDIR/summary.txt"
if ((overall_pass == 0)); then
  exit 1
fi
