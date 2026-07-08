#!/usr/bin/env bash

set -euo pipefail

readonly DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd "$DIR/../.." && pwd)"
readonly OCLDIR="$DIR/../opencl"
readonly OUTDIR="$DIR/results-pocl-nvidia"
readonly POCL_LIBDIR="$ROOT/install/lib"
readonly POCL_LOADER="$POCL_LIBDIR/libOpenCL.so.1"
readonly POCL_SYMLINK_TARGET="libOpenCL.so.2.17.0"
readonly DEFAULT_BENCHMARKS="b+tree backprop bfs cfd dwt2d gaussian heartwall hotspot hotspot3D hybridsort kmeans lavaMD leukocyte lud myocyte nn nw particlefilter pathfinder srad streamcluster"
readonly DEFAULT_NATIVE_PLATFORM=0
readonly DEFAULT_NATIVE_DEVICE=0
readonly DEFAULT_POCL_PLATFORM=0
readonly DEFAULT_POCL_DEVICE=1
readonly DEFAULT_POCL_ARCH="sm_89"

mkdir -p "$OUTDIR"

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

usage() {
  cat <<'EOF'
Usage: run_pocl_nvidia.sh [options]

  --validate                 compare native NVIDIA OpenCL outputs with pocl GPU outputs
  --iterations N             run each backend N times (default: 1)
  --benchmarks "a b c"       space-separated benchmark list
  --native-platform N        native OpenCL platform id (default: 0)
  --native-device N          native OpenCL device id (default: 0)
  --pocl-platform N          pocl platform id (default: 0)
  --pocl-device N            pocl device id (default: 1)
  --pocl-arch ARCH           POCL_CUDA_GPU_ARCH value (default: sm_89)
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

ensure_pocl_loader() {
  if [[ ! -e "$POCL_LIBDIR/$POCL_SYMLINK_TARGET" ]]; then
    echo "Missing PoCL loader target: $POCL_LIBDIR/$POCL_SYMLINK_TARGET" >&2
    return 1
  fi
  if [[ ! -e "$POCL_LOADER" ]]; then
    ln -sf "$POCL_SYMLINK_TARGET" "$POCL_LOADER"
  fi
}

normalize_log() {
  sed -E \
    -e '/^[[:space:]]*$/d' \
    -e '/no version information available/d' \
    -e '/^(Init|MemAlloc|HtoD|DtoH|Exec|Close|Total):/d' \
    -e '/^(Platform|Device):/d' \
    -e '/^WG size /d' \
    -e '/^num_devices = /d' \
    -e '/^Running on: /d' \
    -e '/^Use GPU device$/d' \
    -e '/^Time spent in different stages/d' \
    -e '/^TOTAL TIME:/d' \
    -e '/^[[:space:]]*Kernel[[:space:]]+[0-9]/d' \
    "$1"
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
        "POCL_CUDA_GPU_ARCH=$pocl_arch"
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

  (
    cd "$OCLDIR/$benchmark"
    env "${envs[@]}" "${run_cmd[@]}"
  ) |& tee "$log_file"
  status=${PIPESTATUS[0]}

  copy_outputs "$benchmark" "$runtime" "$iter"
  return "$status"
}

log_has_runtime_error() {
  local log_file=$1
  grep -Eiq \
    '(^ERROR:|Segmentation fault|core dumped|exception|Failed to create|Could not create|CL_INVALID_|No devices available)' \
    "$log_file"
}

compare_text_outputs() {
  local file_a=$1
  local file_b=$2
  local abs_tol=$3
  local rel_tol=$4

  awk -v abs_tol="$abs_tol" -v rel_tol="$rel_tol" '
    function absval(x) { return x < 0 ? -x : x }
    function isnum(x, y) {
      y = tolower(x)
      return y == "nan" || y == "+nan" || y == "-nan" || y == "inf" || y == "+inf" || y == "-inf" \
        || x ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/
    }
    function push_tokens(line, array_name,    tmp, n, i) {
      gsub(/,/, " ", line)
      gsub(/=/, " ", line)
      gsub(/\r/, "", line)
      n = split(line, tmp, /[[:space:]]+/)
      for (i = 1; i <= n; ++i) {
        if (tmp[i] == "")
          continue
        if (array_name == "A")
          A[++count_a] = tmp[i]
        else
          B[++count_b] = tmp[i]
      }
    }
    FNR == NR {
      push_tokens($0, "A")
      next
    }
    {
      push_tokens($0, "B")
    }
    END {
      if (count_a != count_b)
        exit 1
      for (i = 1; i <= count_a; ++i) {
        if (A[i] == B[i])
          continue
        if (!isnum(A[i]) || !isnum(B[i]))
          exit 1
        if (tolower(A[i]) ~ /nan/ && tolower(B[i]) ~ /nan/)
          continue
        av = A[i] + 0
        bv = B[i] + 0
        diff = absval(av - bv)
        scale = absval(av)
        if (absval(bv) > scale)
          scale = absval(bv)
        limit = abs_tol
        if (rel_tol * scale > limit)
          limit = rel_tol * scale
        if (diff > limit)
          exit 1
      }
    }
  ' "$file_a" "$file_b"
}

compare_pgm_outputs() {
  local file_a=$1
  local file_b=$2
  local abs_tol=$3

  awk -v abs_tol="$abs_tol" '
    function absval(x) { return x < 0 ? -x : x }
    function push_tokens(array_name,    i) {
      for (i = 1; i <= NF; ++i)
        if (array_name == "A")
          A[++count_a] = $i
        else
          B[++count_b] = $i
    }
    FNR == NR {
      push_tokens("A")
      next
    }
    {
      push_tokens("B")
    }
    END {
      if (count_a != count_b)
        exit 1
      for (i = 1; i <= 4; ++i) {
        if (A[i] != B[i])
          exit 1
      }
      for (i = 5; i <= count_a; ++i) {
        if (absval((A[i] + 0) - (B[i] + 0)) > abs_tol)
          exit 1
      }
    }
  ' "$file_a" "$file_b"
}

validate_outputs() {
  local benchmark=$1
  local iter=$2
  local outputs="${OUTPUT_FILES[$benchmark]-}"
  local abs_tol="${ABS_TOLERANCES[$benchmark]-0}"
  local rel_tol="${REL_TOLERANCES[$benchmark]-0}"
  local idx=0
  local out_file

  if [[ -n "$outputs" ]]; then
    for out_file in ${outputs//|/ }; do
      local native_file="$OUTDIR/$benchmark.native.$iter.$idx.out"
      local pocl_file="$OUTDIR/$benchmark.pocl.$iter.$idx.out"
      if [[ ! -f "$native_file" || ! -f "$pocl_file" ]]; then
        echo "  validate failed for $benchmark (iter $iter): missing $out_file" >&2
        return 1
      fi
      if [[ "$benchmark" == "srad" ]]; then
        if ! compare_pgm_outputs "$native_file" "$pocl_file" 1; then
          echo "  validate failed for $benchmark (iter $iter): $out_file mismatch" >&2
          return 1
        fi
      elif [[ "$abs_tol" != 0 || "$rel_tol" != 0 ]]; then
        if ! compare_text_outputs "$native_file" "$pocl_file" "$abs_tol" "$rel_tol"; then
          echo "  validate failed for $benchmark (iter $iter): $out_file mismatch" >&2
          return 1
        fi
      else
        if ! cmp -s "$native_file" "$pocl_file"; then
          echo "  validate failed for $benchmark (iter $iter): $out_file mismatch" >&2
          return 1
        fi
      fi
      idx=$((idx + 1))
    done
    return 0
  fi

  if ! diff -u \
    <(normalize_log "$OUTDIR/$benchmark.native.$iter.log") \
    <(normalize_log "$OUTDIR/$benchmark.pocl.$iter.log") \
    > "$OUTDIR/$benchmark.validate.$iter.diff"; then
    echo "  validate failed for $benchmark (iter $iter): normalized log mismatch" >&2
    return 1
  fi

  return 0
}

ensure_pocl_loader

for benchmark in $benchmarks; do
  : > "$OUTDIR/$benchmark.txt"
  echo "$(date) # running $benchmark"

  if [[ ! -x "$OCLDIR/$benchmark/run" ]]; then
    chmod +x "$OCLDIR/$benchmark/run" 2>/dev/null || true
  fi

  if [[ ! -f "$OCLDIR/$benchmark/run" ]]; then
    echo "  missing run script for $benchmark" >&2
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
    continue
  fi

  if ! build_benchmark "$benchmark"; then
    echo "  build failed for $benchmark" >&2
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
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
  else
    printf "validate=%d pass=%s\n\n" "$validate" "NO" >> "$OUTDIR/$benchmark.txt"
  fi

  echo
done
