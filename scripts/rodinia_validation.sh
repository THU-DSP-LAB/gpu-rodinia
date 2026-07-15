#!/usr/bin/env bash

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

validate_btree_output() {
  local file=$1

  awk '
    BEGIN {
      split("840187 394382 783099 798440 911647 197551 335222 768229 277774 553970", expect_j, " ")
      split("477397 628870 364784 513400 952229 916195 635711 717296 141602 606968", expect_k, " ")
      mode = ""
      j_seen = 0
      k_seen = 0
      j_count = 0
      k_count = 0
    }
    /^[[:space:]]*\*+[[:space:]]*command:[[:space:]]*j[[:space:]]+count=10,[[:space:]]*rSize=10[[:space:]]*$/ {
      mode = "j"
      next
    }
    /^[[:space:]]*\*+[[:space:]]*command:[[:space:]]*k[[:space:]]+count=10[[:space:]]*$/ {
      mode = "k"
      next
    }
    mode == "j" && $1 ~ /^[0-9]+$/ {
      idx = $1 + 1
      if (idx < 1 || idx > 10 || $2 != expect_j[idx] || $3 != 11)
        exit 1
      j_count++
      if (j_count == 10) {
        j_seen = 1
        mode = ""
      }
      next
    }
    mode == "k" && $1 ~ /^[0-9]+$/ {
      idx = $1 + 1
      if (idx < 1 || idx > 10 || $2 != expect_k[idx])
        exit 1
      k_count++
      if (k_count == 10) {
        k_seen = 1
        mode = ""
      }
      next
    }
    END {
      if (!j_seen || !k_seen || j_count != 10 || k_count != 10)
        exit 1
    }
  ' "$file"
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
    if [[ "$benchmark" == "b+tree" ]]; then
      local pocl_file="$OUTDIR/$benchmark.pocl.$iter.0.out"
      if ! validate_btree_output "$pocl_file"; then
        echo "  validate failed for $benchmark (iter $iter): pocl output does not match expected queries" >&2
        return 1
      fi
      return 0
    fi

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
      elif ! cmp -s "$native_file" "$pocl_file"; then
        echo "  validate failed for $benchmark (iter $iter): $out_file mismatch" >&2
        return 1
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
}
