#include "backprop_result.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/rodinia_verify.h"

enum {
  REFERENCE_MAGIC = 0x42505248,
  REFERENCE_VERSION = 2,
  FIRST_MODEL_UNIT = 1
};

static const uint64_t FNV1A_OFFSET = 1469598103934665603ULL;
static const uint64_t FNV1A_PRIME = 1099511628211ULL;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t input_n;
  uint32_t hidden_n;
  uint32_t output_n;
  float output_error;
  float hidden_error;
  uint64_t hidden_units_hash;
  uint64_t output_units_hash;
  uint64_t input_weights_hash;
  uint64_t hidden_weights_hash;
} ReferenceHeader;

typedef struct {
  uint64_t hidden_units;
  uint64_t output_units;
  uint64_t input_weights;
  uint64_t hidden_weights;
} ResultHashes;

static int open_file_failed(const char *path)
{
  fprintf(stderr, "Failed to open '%s': %s\n", path, strerror(errno));
  return -1;
}

static int read_or_write_failed(const char *action, const char *label, const char *path)
{
  fprintf(stderr, "Failed to %s %s in '%s'\n", action, label, path);
  return -1;
}

static uint64_t fnv1a_update(uint64_t hash, const void *data, size_t size)
{
  const unsigned char *bytes = (const unsigned char *) data;

  for (size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= FNV1A_PRIME;
  }

  return hash;
}

static uint64_t hash_vector(const float *values, int count)
{
  if (count <= 0) {
    return FNV1A_OFFSET;
  }

  return fnv1a_update(FNV1A_OFFSET, values, (size_t) count * sizeof(float));
}

static uint64_t hash_vector_from_index(const float *values, int count, int first_index)
{
  if (first_index >= count) {
    return FNV1A_OFFSET;
  }

  return hash_vector(values + first_index, count - first_index);
}

static uint64_t hash_matrix(float **matrix, int rows, int cols)
{
  uint64_t hash = FNV1A_OFFSET;

  for (int row_index = 0; row_index < rows; ++row_index) {
    hash = fnv1a_update(hash, matrix[row_index], (size_t) cols * sizeof(float));
  }

  return hash;
}

static ResultHashes compute_hashes(const BPNN *net)
{
  const ResultHashes hashes = {
      hash_vector(net->hidden_units, net->hidden_n + 1),
      hash_vector_from_index(net->output_units, net->output_n + 1, FIRST_MODEL_UNIT),
      hash_matrix(net->input_weights, net->input_n + 1, net->hidden_n + 1),
      hash_matrix(net->hidden_weights, net->hidden_n + 1, net->output_n + 1)};

  return hashes;
}

static int write_header(FILE *file, const ReferenceHeader *header, const char *path)
{
  if (fwrite(header, sizeof(*header), 1, file) != 1) {
    return read_or_write_failed("write", "reference header", path);
  }
  return 0;
}

static int read_header(FILE *file, ReferenceHeader *header, const char *path)
{
  if (fread(header, sizeof(*header), 1, file) != 1) {
    return read_or_write_failed("read", "reference header", path);
  }
  return 0;
}

static int verify_scalar(
    const char *label,
    float actual,
    float expected,
    float tolerance)
{
  const float abs_diff = fabsf(actual - expected);

  if (abs_diff <= tolerance) {
    return 0;
  }

  fprintf(
      stderr,
      "Reference mismatch for %s: actual=%0.8f expected=%0.8f abs_diff=%0.8f tolerance=%0.8f\n",
      label,
      actual,
      expected,
      abs_diff,
      tolerance);
  return -1;
}

static int verify_hash(const char *label, uint64_t actual, uint64_t expected)
{
  if (actual == expected) {
    return 0;
  }

  fprintf(
      stderr,
      "Reference mismatch for %s: actual=%016llx expected=%016llx\n",
      label,
      (unsigned long long) actual,
      (unsigned long long) expected);
  return -1;
}

static int verify_dimensions(const BPNN *net, const ReferenceHeader *header)
{
  if (header->input_n != (uint32_t) net->input_n ||
      header->hidden_n != (uint32_t) net->hidden_n ||
      header->output_n != (uint32_t) net->output_n) {
    fprintf(
        stderr,
        "Reference dimensions mismatch: actual=%dx%dx%d expected=%ux%ux%u\n",
        net->input_n,
        net->hidden_n,
        net->output_n,
        header->input_n,
        header->hidden_n,
        header->output_n);
    return -1;
  }

  return 0;
}

static int verify_header(const BPNN *net, const ReferenceHeader *header)
{
  if (header->magic != REFERENCE_MAGIC) {
    fprintf(stderr, "Invalid reference magic: 0x%08x\n", header->magic);
    return -1;
  }

  if (header->version != REFERENCE_VERSION) {
    fprintf(
        stderr,
        "Unsupported reference version: actual=%u expected=%u\n",
        header->version,
        REFERENCE_VERSION);
    return -1;
  }

  return verify_dimensions(net, header);
}

int bpnn_save_reference(
    const BPNN *net,
    float output_error,
    float hidden_error,
    const char *path)
{
  const ResultHashes hashes = compute_hashes(net);
  const ReferenceHeader header = {
      REFERENCE_MAGIC,
      REFERENCE_VERSION,
      (uint32_t) net->input_n,
      (uint32_t) net->hidden_n,
      (uint32_t) net->output_n,
      output_error,
      hidden_error,
      hashes.hidden_units,
      hashes.output_units,
      hashes.input_weights,
      hashes.hidden_weights};
  FILE *file = fopen(path, "wb");

  if (file == NULL) {
    return open_file_failed(path);
  }

  if (write_header(file, &header, path) != 0) {
    fclose(file);
    return -1;
  }

  fclose(file);
  printf("Saved reference result to '%s'\n", path);
  return 0;
}

int bpnn_verify_reference(
    const BPNN *net,
    float output_error,
    float hidden_error,
    const char *path,
    float tolerance)
{
  ReferenceHeader header;
  FILE *file = fopen(path, "rb");
  ResultHashes hashes;

  if (file == NULL) {
    return open_file_failed(path);
  }

  if (read_header(file, &header, path) != 0 || verify_header(net, &header) != 0) {
    fclose(file);
    return -1;
  }
  fclose(file);

  hashes = compute_hashes(net);
  if (verify_scalar("output_error", output_error, header.output_error, tolerance) != 0 ||
      verify_scalar("hidden_error", hidden_error, header.hidden_error, tolerance) != 0 ||
      verify_hash("hidden_units", hashes.hidden_units, header.hidden_units_hash) != 0 ||
      verify_hash("output_units", hashes.output_units, header.output_units_hash) != 0 ||
      verify_hash("input_weights", hashes.input_weights, header.input_weights_hash) != 0 ||
      verify_hash("hidden_weights", hashes.hidden_weights, header.hidden_weights_hash) != 0) {
    return -1;
  }

  printf("Backprop reference verification matched '%s' with tolerance %0.8f\n", path, tolerance);
  return rodinia_print_pass("Backprop reference verification");
}
