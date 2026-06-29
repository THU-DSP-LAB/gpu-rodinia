#ifndef DWT_REFERENCE_H
#define DWT_REFERENCE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "common.h"
#include "../../common/rodinia_verify.h"

#define DWT2D_REFERENCE_MAGIC "GPIDL_RODINIA_DWT2D_REFERENCE"
#define DWT2D_REFERENCE_VERSION 1
#define DWT2D_MAX_COMPONENTS 4

typedef struct {
    uint64_t hash;
    uint64_t bytes;
} Dwt2DFileDigest;

typedef struct {
    char input_name[256];
    uint64_t input_hash;
    uint64_t input_bytes;
    int width;
    int height;
    int components;
    int bit_depth;
    int levels;
    int forward;
    int dwt97;
    int write_visual;
    int sample_size;
    char sample_type[16];
    uint64_t component_values;
    uint64_t component_hashes[DWT2D_MAX_COMPONENTS];
} Dwt2DReference;

typedef struct {
    const char *save_path;
    const char *verify_path;
    Dwt2DReference reference;
} Dwt2DReferenceContext;

static uint64_t dwt2d_fnv1a_update(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static const char *dwt2d_basename(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash == NULL ? path : slash + 1;
}

static int dwt2d_compute_file_digest(const char *path, Dwt2DFileDigest *digest)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open DWT2D input for digest: %s\n", path);
        return -1;
    }

    digest->hash = 1469598103934665603ULL;
    digest->bytes = 0;
    unsigned char buffer[65536];
    while (1) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        digest->hash = dwt2d_fnv1a_update(digest->hash, buffer, count);
        digest->bytes += count;
        if (count != sizeof(buffer)) {
            if (ferror(file)) {
                fprintf(stderr, "Failed reading DWT2D input for digest: %s\n", path);
                fclose(file);
                return -1;
            }
            break;
        }
    }
    fclose(file);
    return 0;
}

static void dwt2d_init_reference(
    Dwt2DReference *reference,
    const char *input_path,
    const Dwt2DFileDigest *digest,
    int width,
    int height,
    int components,
    int bit_depth,
    int levels,
    int forward,
    int dwt97,
    int write_visual,
    int sample_size,
    const char *sample_type)
{
    memset(reference, 0, sizeof(*reference));
    snprintf(reference->input_name, sizeof(reference->input_name), "%s", dwt2d_basename(input_path));
    reference->input_hash = digest->hash;
    reference->input_bytes = digest->bytes;
    reference->width = width;
    reference->height = height;
    reference->components = components;
    reference->bit_depth = bit_depth;
    reference->levels = levels;
    reference->forward = forward;
    reference->dwt97 = dwt97;
    reference->write_visual = write_visual;
    reference->sample_size = sample_size;
    snprintf(reference->sample_type, sizeof(reference->sample_type), "%s", sample_type);
    reference->component_values = (uint64_t)width * (uint64_t)height;
}

template <typename T>
static int dwt2d_record_component_hash(
    Dwt2DReferenceContext *context,
    int component_index,
    T *component_cuda,
    int samples)
{
    if (context == NULL) {
        return 0;
    }
    if (component_index < 0 || component_index >= DWT2D_MAX_COMPONENTS) {
        fprintf(stderr, "Invalid DWT2D component index: %d\n", component_index);
        return -1;
    }

    std::vector<T> values(samples);
    cudaMemcpy(values.data(), component_cuda, (size_t)samples * sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheckError("Copy DWT2D reference output to host");
    context->reference.component_hashes[component_index] =
        dwt2d_fnv1a_update(1469598103934665603ULL, values.data(), (size_t)samples * sizeof(T));
    return 0;
}

static int dwt2d_save_reference(const char *path, const Dwt2DReference *reference)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot open DWT2D reference for write: %s\n", path);
        return -1;
    }

    fprintf(file, "%s %d\n", DWT2D_REFERENCE_MAGIC, DWT2D_REFERENCE_VERSION);
    fprintf(file, "input_name %s\n", reference->input_name);
    fprintf(file, "input_hash %016llx\n", (unsigned long long)reference->input_hash);
    fprintf(file, "input_bytes %llu\n", (unsigned long long)reference->input_bytes);
    fprintf(file, "width %d\n", reference->width);
    fprintf(file, "height %d\n", reference->height);
    fprintf(file, "components %d\n", reference->components);
    fprintf(file, "bit_depth %d\n", reference->bit_depth);
    fprintf(file, "levels %d\n", reference->levels);
    fprintf(file, "forward %d\n", reference->forward);
    fprintf(file, "dwt97 %d\n", reference->dwt97);
    fprintf(file, "write_visual %d\n", reference->write_visual);
    fprintf(file, "sample_size %d\n", reference->sample_size);
    fprintf(file, "sample_type %s\n", reference->sample_type);
    fprintf(file, "component_values %llu\n", (unsigned long long)reference->component_values);
    for (int i = 0; i < reference->components; i++) {
        fprintf(file, "component_%d_hash %016llx\n", i, (unsigned long long)reference->component_hashes[i]);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed closing DWT2D reference: %s\n", path);
        return -1;
    }
    printf("Saved DWT2D reference to '%s'\n", path);
    return 0;
}

static int dwt2d_scan_field(FILE *file, const char *label, const char *format, void *value)
{
    char actual_label[64];
    if (fscanf(file, "%63s", actual_label) != 1 ||
        strcmp(actual_label, label) != 0 ||
        fscanf(file, format, value) != 1) {
        fprintf(stderr, "Invalid DWT2D reference field: %s\n", label);
        return -1;
    }
    return 0;
}

static int dwt2d_read_reference(const char *path, Dwt2DReference *reference)
{
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open DWT2D reference for read: %s\n", path);
        return -1;
    }

    char magic[128];
    int version = 0;
    if (fscanf(file, "%127s %d", magic, &version) != 2 ||
        strcmp(magic, DWT2D_REFERENCE_MAGIC) != 0 ||
        version != DWT2D_REFERENCE_VERSION) {
        fprintf(stderr, "Invalid DWT2D reference header: %s\n", path);
        fclose(file);
        return -1;
    }

    unsigned long long ull_value = 0;
    int failed = 0;
    memset(reference, 0, sizeof(*reference));
    failed |= dwt2d_scan_field(file, "input_name", "%255s", reference->input_name);
    failed |= dwt2d_scan_field(file, "input_hash", "%llx", &ull_value);
    reference->input_hash = ull_value;
    failed |= dwt2d_scan_field(file, "input_bytes", "%llu", &ull_value);
    reference->input_bytes = ull_value;
    failed |= dwt2d_scan_field(file, "width", "%d", &reference->width);
    failed |= dwt2d_scan_field(file, "height", "%d", &reference->height);
    failed |= dwt2d_scan_field(file, "components", "%d", &reference->components);
    if (failed == 0 &&
        (reference->components < 1 || reference->components > DWT2D_MAX_COMPONENTS)) {
        fprintf(stderr, "Invalid DWT2D reference component count: %d\n", reference->components);
        failed = -1;
    }
    failed |= dwt2d_scan_field(file, "bit_depth", "%d", &reference->bit_depth);
    failed |= dwt2d_scan_field(file, "levels", "%d", &reference->levels);
    failed |= dwt2d_scan_field(file, "forward", "%d", &reference->forward);
    failed |= dwt2d_scan_field(file, "dwt97", "%d", &reference->dwt97);
    failed |= dwt2d_scan_field(file, "write_visual", "%d", &reference->write_visual);
    failed |= dwt2d_scan_field(file, "sample_size", "%d", &reference->sample_size);
    failed |= dwt2d_scan_field(file, "sample_type", "%15s", reference->sample_type);
    failed |= dwt2d_scan_field(file, "component_values", "%llu", &ull_value);
    reference->component_values = ull_value;

    for (int i = 0; failed == 0 && i < reference->components; i++) {
        char label[64];
        snprintf(label, sizeof(label), "component_%d_hash", i);
        failed |= dwt2d_scan_field(file, label, "%llx", &ull_value);
        reference->component_hashes[i] = ull_value;
    }
    fclose(file);
    return failed == 0 ? 0 : -1;
}

static int dwt2d_compare_reference(const Dwt2DReference *actual, const Dwt2DReference *expected)
{
    int failed = 0;
    failed |= strcmp(actual->input_name, expected->input_name) != 0;
    failed |= actual->input_hash != expected->input_hash;
    failed |= actual->input_bytes != expected->input_bytes;
    failed |= actual->width != expected->width;
    failed |= actual->height != expected->height;
    failed |= actual->components != expected->components;
    failed |= actual->bit_depth != expected->bit_depth;
    failed |= actual->levels != expected->levels;
    failed |= actual->forward != expected->forward;
    failed |= actual->dwt97 != expected->dwt97;
    failed |= actual->write_visual != expected->write_visual;
    failed |= actual->sample_size != expected->sample_size;
    failed |= strcmp(actual->sample_type, expected->sample_type) != 0;
    failed |= actual->component_values != expected->component_values;
    for (int i = 0; i < actual->components && i < DWT2D_MAX_COMPONENTS; i++) {
        failed |= actual->component_hashes[i] != expected->component_hashes[i];
    }
    return failed ? -1 : 0;
}

static int dwt2d_finish_reference(Dwt2DReferenceContext *context)
{
    if (context == NULL) {
        return 0;
    }
    if (context->save_path != NULL) {
        return dwt2d_save_reference(context->save_path, &context->reference);
    }

    Dwt2DReference expected;
    if (dwt2d_read_reference(context->verify_path, &expected) != 0 ||
        dwt2d_compare_reference(&context->reference, &expected) != 0) {
        rodinia_print_fail("DWT2D reference verification");
        return -1;
    }
    return rodinia_print_pass("DWT2D reference verification");
}

#endif
