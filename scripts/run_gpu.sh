#!/usr/bin/env bash

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCLDIR=$DIR/../opencl
OUTDIR=$DIR/results-gpu
mkdir -p "$OUTDIR"

bm="backprop bfs cfd gaussian hotspot hotspot3D hybridsort lud \
    nn nw pathfinder srad streamcluster"

declare -A OUTPUT_FILES=(
    [backprop]=backprop_result.txt
    [cfd]='density.txt|momentum.txt|density_energy.txt'
    [bfs]=bfs_result.txt
    [hotspot]=output.txt
    [hotspot3D]=output.txt
    [gaussian]=gaussian_result.txt
    [hybridsort]=hybridoutput.txt
    [lud]=lud_verify.txt
    [nn]=nn_result.txt
    [nw]=nw_result.txt
    [pathfinder]=pathfinder_result.txt
    [srad]=output/image_out.pgm
    [streamcluster]=output.txt
)

declare -A PASS_MARKERS=(
    [bfs]='--cambine:passed:-)'
)

declare -A VALIDATE_RUN_ARGS=(
    [backprop]='--validate'
    [bfs]='--validate'
    [gaussian]='--validate'
    [hybridsort]='--validate'
    [lud]='-v'
    [nn]='--validate'
    [nw]='--traceback'
    [pathfinder]='--validate'
)

declare -A VALIDATE_MAKE_ARGS=(
    [hybridsort]='VERIFY=1 OUTPUT=1'
)

validate=0
validate_mode=semantic
iterations=1
platform_cpu=0
platform_cuda=0

usage() {
  cat <<'EOF'
Usage: run_gpu.sh [--validate] [--iterations N]

  --validate       compare GPU results with CPU reference for each benchmark
  --validate-mode  semantic|log (default: semantic)
                  semantic: prefer output-marker/file semantic checks, then fallback log
                  log: force normalized log diff for all validate cases
  --iterations N   run count per benchmark in each mode (default 1)
EOF
}

while (("$#")); do
  case "${1:-}" in
    --validate)
      validate=1
      shift
      ;;
    --validate-mode)
      validate_mode="${2:-}"
      if [[ "$validate_mode" != "semantic" && "$validate_mode" != "log" ]]; then
        echo "Invalid --validate-mode: $validate_mode" >&2
        usage
        exit 1
      fi
      shift 2
      ;;
    --iterations)
      iterations="${2:-}"
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

run_once() {
  local b=$1
  local target=$2
  local iter=$3
  local log_file=$4
  local outputs=${OUTPUT_FILES[$b]-}
  local -a run_cmd=(./run)
  local -a run_extra_args=()
  local run_status=0
  for out_file in ${outputs//|/ }; do
    rm -f "$OCLDIR/$b/$out_file"
  done
  local -a envs=(
    "POCL_BUILDING=1"
    "POCL_DEVICES=${target}"
  )

  if ((validate)); then
    local run_args=${VALIDATE_RUN_ARGS[$b]-}
    [ -n "$run_args" ] && read -r -a run_extra_args <<< "$run_args"
    run_cmd+=("${run_extra_args[@]}")
  fi

  if [[ $target == "CPU" ]]; then
    local platform=$platform_cpu
    run_cmd+=(-p "$platform" -d 0)
    (cd "$OCLDIR/$b" && env "${envs[@]}" "${run_cmd[@]}") \
      |& tee "$log_file"
  else
    local platform=$platform_cuda
    run_cmd+=(-p "$platform" -d 0)
    (cd "$OCLDIR/$b" && env "${envs[@]}" "${run_cmd[@]}") \
      |& tee "$log_file"
  fi
  run_status=${PIPESTATUS[0]}

  local idx=0
  for out_file in ${outputs//|/ }; do
    if [ -f "$OCLDIR/$b/$out_file" ]; then
      cp "$OCLDIR/$b/$out_file" "$OUTDIR/$b.${target,,}.$iter.$idx.out"
    fi
    ((idx += 1))
  done
  return "$run_status"
}

normalize_log() {
  sed -E \
    -e '/^[[:space:]]*$/d' \
    -e '/(Init|MemAlloc|HtoD|DtoH|Exec|Close|Total|GPU|Kernel|KernelTime|time|Timing|Runtime|Elapsed)/d' \
    -e '/\[[0-9.]+MB\]|\[[0-9.]+KB\]|\[[0-9]+MB\]|\[[0-9]+KB\]/d' \
    "$1"
}

has_fatal() {
  local log_file=$1
  grep -Eqi \
    'Segmentation fault|core dumped|illegal instruction|Aborted|Bus error' \
    "$log_file"
}

validate_by_semantic() {
  local b=$1
  local iter=$2
  local marker=${PASS_MARKERS[$b]-}
  local cpu_log="$OUTDIR/$b.cpu.$iter.log"
  local cuda_log="$OUTDIR/$b.cuda.$iter.log"

  if has_fatal "$cpu_log" || has_fatal "$cuda_log"; then
    echo "  validate failed by fatal log marker for $b (iter $iter)" >&2
    return 1
  fi

  if [ -n "$marker" ]; then
    if ! grep -qF -- "$marker" "$cpu_log" || ! grep -qF -- "$marker" "$cuda_log"; then
      echo "  validate failed by marker check for $b (iter $iter)" >&2
      return 1
    fi
  fi

  if [ -n "${OUTPUT_FILES[$b]-}" ]; then
    local idx=0
    for out_file in ${OUTPUT_FILES[$b]//|/ }; do
      if [ ! -f "$OUTDIR/$b.cpu.$iter.$idx.out" ] || [ ! -f "$OUTDIR/$b.cuda.$iter.$idx.out" ]; then
        echo "  validate failed by output file compare for $b (iter $iter): missing output file $out_file" >&2
        return 1
      fi
      if ! cmp -s "$OUTDIR/$b.cpu.$iter.$idx.out" "$OUTDIR/$b.cuda.$iter.$idx.out"; then
        echo "  validate failed by output file compare for $b (iter $iter): $out_file mismatch" >&2
        return 1
      fi
      ((idx += 1))
    done
  elif [[ "$validate_mode" == "log" ]]; then
    if ! diff -u \
      <(normalize_log "$cpu_log") \
      <(normalize_log "$cuda_log") \
      > "$OUTDIR/$b.validate.$iter.diff"; then
      echo "  validate failed by log diff for $b (iter $iter)" >&2
      return 1
    fi
  fi

  if [[ "$validate_mode" == "semantic" ]]; then
    # No explicit semantic criterion yet for this benchmark:
    # keep a conservative log fallback only for this case.
    if [ -z "${PASS_MARKERS[$b]-}" ] && [ -z "${OUTPUT_FILES[$b]-}" ]; then
      if ! diff -u \
        <(normalize_log "$cpu_log") \
        <(normalize_log "$cuda_log") \
        > "$OUTDIR/$b.validate.$iter.diff"; then
        echo "  validate failed by log diff fallback for $b (iter $iter)" >&2
        return 1
      fi
    fi
  fi

  return 0
}

for b in $bm; do
  echo -n > "$OUTDIR/$b.txt"
  echo "$(date) # running $b"
  cd "$OCLDIR/$b"
  make_args=()
  read -r -a make_extra <<< "${VALIDATE_MAKE_ARGS[$b]-}"
  if ((validate)) && [ "${#make_extra[@]}" -gt 0 ]; then
    make_args=("${make_extra[@]}")
  fi
  make clean TYPE=GPU >/dev/null 2>&1 || true
  if ! make -j$(nproc) TYPE=GPU CC='g++ -std=gnu++03' "${make_args[@]}" |& tee "$OUTDIR/$b.make.log"; then
    pass=0
    echo "  build failed for $b" >&2
    cd "$DIR"
    printf "validate=%d pass=%s\n" "$validate" "NO" >> "$OUTDIR/$b.txt"
    echo >> "$OUTDIR/$b.txt"
    continue
  fi

  pass=1
  if ((validate)); then
    for idx in $(seq 1 "$iterations"); do
    if ! run_once "$b" "CPU" "$idx" "$OUTDIR/$b.cpu.$idx.log"; then
        pass=0
        echo "  CPU run failed for $b (iter $idx)" >&2
      fi
    done
    for idx in $(seq 1 "$iterations"); do
      if ! run_once "$b" "CUDA" "$idx" "$OUTDIR/$b.cuda.$idx.log"; then
        pass=0
        echo "  CUDA run failed for $b (iter $idx)" >&2
      elif ! validate_by_semantic "$b" "$idx"; then
        pass=0
      fi
    done
  else
    for idx in $(seq 1 "$iterations"); do
      if ! run_once "$b" "CUDA" "$idx" "$OUTDIR/$b.cuda.$idx.log"; then
        pass=0
        echo "  CUDA run failed for $b (iter $idx)" >&2
      fi
    done
  fi

  if ((pass)); then
    printf "validate=%d pass=%s\n" "$validate" "YES" >> "$OUTDIR/$b.txt"
  else
    printf "validate=%d pass=%s\n" "$validate" "NO" >> "$OUTDIR/$b.txt"
  fi

  echo >> "$OUTDIR/$b.txt"
  cd "$DIR"
  echo
done
