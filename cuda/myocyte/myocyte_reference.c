#include <stdint.h>
#include "../../common/rodinia_verify.h"

typedef struct MyocyteFileDigest {
	unsigned long long hash;
	unsigned long long bytes;
	unsigned long long lines;
} MyocyteFileDigest;

typedef struct MyocyteReference {
	int xmax;
	int workload;
	int mode;
	int equations;
	int parameters;
	char y_path[256];
	unsigned long long y_hash;
	unsigned long long y_bytes;
	unsigned long long y_lines;
	char params_path[256];
	unsigned long long params_hash;
	unsigned long long params_bytes;
	unsigned long long params_lines;
	char output_path[256];
	unsigned long long output_hash;
	unsigned long long output_bytes;
	unsigned long long output_lines;
} MyocyteReference;

static const unsigned long long FNV_OFFSET_BASIS = 1469598103934665603ULL;
static const unsigned long long FNV_PRIME = 1099511628211ULL;
static const char *MYOCYTE_REFERENCE_MAGIC = "GPIDL_RODINIA_MYOCYTE_REFERENCE";
static const char *MYOCYTE_Y_PATH = "../../data/myocyte/y.txt";
static const char *MYOCYTE_PARAMS_PATH = "../../data/myocyte/params.txt";
static const char *MYOCYTE_OUTPUT_PATH = "output.txt";

static void printUsage(const char *program_name)
{
	fprintf(stderr, "Usage: %s <xmax> <workload> <mode> [--save-reference <path>|--verify-reference <path>]\n", program_name);
}

static int copyBoundedString(char *destination, size_t destination_size, const char *source)
{
	if (strlen(source) >= destination_size) {
		fprintf(stderr, "Reference path is too long: %s\n", source);
		return 1;
	}
	strcpy(destination, source);
	return 0;
}

static int computeFileDigest(const char *path, MyocyteFileDigest *digest)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "Cannot open file for digest: %s\n", path);
		return 1;
	}

	digest->hash = FNV_OFFSET_BASIS;
	digest->bytes = 0;
	digest->lines = 0;
	unsigned char buffer[65536];
	while (1) {
		size_t count = fread(buffer, 1, sizeof(buffer), file);
		for (size_t index = 0; index < count; ++index) {
			digest->hash ^= buffer[index];
			digest->hash *= FNV_PRIME;
			if (buffer[index] == '\n') {
				digest->lines += 1;
			}
		}
		digest->bytes += count;
		if (count != sizeof(buffer)) {
			if (ferror(file)) {
				fprintf(stderr, "Failed reading file for digest: %s\n", path);
				fclose(file);
				return 1;
			}
			break;
		}
	}

	fclose(file);
	return 0;
}

static int populateReference(int xmax, int workload, int mode, MyocyteReference *reference)
{
	MyocyteFileDigest y_digest;
	MyocyteFileDigest params_digest;
	MyocyteFileDigest output_digest;
	if (computeFileDigest(MYOCYTE_Y_PATH, &y_digest) != 0 ||
		computeFileDigest(MYOCYTE_PARAMS_PATH, &params_digest) != 0 ||
		computeFileDigest(MYOCYTE_OUTPUT_PATH, &output_digest) != 0) {
		return 1;
	}

	reference->xmax = xmax;
	reference->workload = workload;
	reference->mode = mode;
	reference->equations = EQUATIONS;
	reference->parameters = PARAMETERS;
	if (copyBoundedString(reference->y_path, sizeof(reference->y_path), MYOCYTE_Y_PATH) != 0 ||
		copyBoundedString(reference->params_path, sizeof(reference->params_path), MYOCYTE_PARAMS_PATH) != 0 ||
		copyBoundedString(reference->output_path, sizeof(reference->output_path), MYOCYTE_OUTPUT_PATH) != 0) {
		return 1;
	}
	reference->y_hash = y_digest.hash;
	reference->y_bytes = y_digest.bytes;
	reference->y_lines = y_digest.lines;
	reference->params_hash = params_digest.hash;
	reference->params_bytes = params_digest.bytes;
	reference->params_lines = params_digest.lines;
	reference->output_hash = output_digest.hash;
	reference->output_bytes = output_digest.bytes;
	reference->output_lines = output_digest.lines;
	return 0;
}

static int writeReference(const char *path, const MyocyteReference *reference)
{
	FILE *file = fopen(path, "w");
	if (file == NULL) {
		fprintf(stderr, "Cannot open reference for writing: %s\n", path);
		return 1;
	}

	fprintf(file, "%s 1\n", MYOCYTE_REFERENCE_MAGIC);
	fprintf(file, "xmax %d\n", reference->xmax);
	fprintf(file, "workload %d\n", reference->workload);
	fprintf(file, "mode %d\n", reference->mode);
	fprintf(file, "equations %d\n", reference->equations);
	fprintf(file, "parameters %d\n", reference->parameters);
	fprintf(file, "y_path %s\n", reference->y_path);
	fprintf(file, "y_hash %016llx\n", reference->y_hash);
	fprintf(file, "y_bytes %llu\n", reference->y_bytes);
	fprintf(file, "y_lines %llu\n", reference->y_lines);
	fprintf(file, "params_path %s\n", reference->params_path);
	fprintf(file, "params_hash %016llx\n", reference->params_hash);
	fprintf(file, "params_bytes %llu\n", reference->params_bytes);
	fprintf(file, "params_lines %llu\n", reference->params_lines);
	fprintf(file, "output_path %s\n", reference->output_path);
	fprintf(file, "output_hash %016llx\n", reference->output_hash);
	fprintf(file, "output_bytes %llu\n", reference->output_bytes);
	fprintf(file, "output_lines %llu\n", reference->output_lines);

	if (fclose(file) != 0) {
		fprintf(stderr, "Failed closing reference after writing: %s\n", path);
		return 1;
	}
	return 0;
}

static int scanString(FILE *file, const char *expected_label, char *value, size_t value_size)
{
	char label[128];
	if (fscanf(file, "%127s %255s", label, value) != 2 || strcmp(label, expected_label) != 0) {
		fprintf(stderr, "Invalid myocyte reference field: %s\n", expected_label);
		return 1;
	}
	value[value_size - 1] = 0;
	return 0;
}

static int scanInt(FILE *file, const char *expected_label, int *value)
{
	char label[128];
	if (fscanf(file, "%127s %d", label, value) != 2 || strcmp(label, expected_label) != 0) {
		fprintf(stderr, "Invalid myocyte reference field: %s\n", expected_label);
		return 1;
	}
	return 0;
}

static int scanHexULL(FILE *file, const char *expected_label, unsigned long long *value)
{
	char label[128];
	if (fscanf(file, "%127s %llx", label, value) != 2 || strcmp(label, expected_label) != 0) {
		fprintf(stderr, "Invalid myocyte reference field: %s\n", expected_label);
		return 1;
	}
	return 0;
}

static int scanDecimalULL(FILE *file, const char *expected_label, unsigned long long *value)
{
	char label[128];
	if (fscanf(file, "%127s %llu", label, value) != 2 || strcmp(label, expected_label) != 0) {
		fprintf(stderr, "Invalid myocyte reference field: %s\n", expected_label);
		return 1;
	}
	return 0;
}

static int readReference(const char *path, MyocyteReference *reference)
{
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		fprintf(stderr, "Cannot open reference for reading: %s\n", path);
		return 1;
	}

	char magic[128];
	int version = 0;
	if (fscanf(file, "%127s %d", magic, &version) != 2 ||
		strcmp(magic, MYOCYTE_REFERENCE_MAGIC) != 0 || version != 1) {
		fprintf(stderr, "Invalid myocyte reference header: %s\n", path);
		fclose(file);
		return 1;
	}

	int failed = 0;
	failed |= scanInt(file, "xmax", &reference->xmax);
	failed |= scanInt(file, "workload", &reference->workload);
	failed |= scanInt(file, "mode", &reference->mode);
	failed |= scanInt(file, "equations", &reference->equations);
	failed |= scanInt(file, "parameters", &reference->parameters);
	failed |= scanString(file, "y_path", reference->y_path, sizeof(reference->y_path));
	failed |= scanHexULL(file, "y_hash", &reference->y_hash);
	failed |= scanDecimalULL(file, "y_bytes", &reference->y_bytes);
	failed |= scanDecimalULL(file, "y_lines", &reference->y_lines);
	failed |= scanString(file, "params_path", reference->params_path, sizeof(reference->params_path));
	failed |= scanHexULL(file, "params_hash", &reference->params_hash);
	failed |= scanDecimalULL(file, "params_bytes", &reference->params_bytes);
	failed |= scanDecimalULL(file, "params_lines", &reference->params_lines);
	failed |= scanString(file, "output_path", reference->output_path, sizeof(reference->output_path));
	failed |= scanHexULL(file, "output_hash", &reference->output_hash);
	failed |= scanDecimalULL(file, "output_bytes", &reference->output_bytes);
	failed |= scanDecimalULL(file, "output_lines", &reference->output_lines);
	fclose(file);
	return failed != 0;
}

static int reportIntMismatch(const char *name, int actual, int expected)
{
	if (actual == expected) {
		return 0;
	}
	fprintf(stderr, "Myocyte reference mismatch for %s: actual=%d expected=%d\n", name, actual, expected);
	return 1;
}

static int reportStringMismatch(const char *name, const char *actual, const char *expected)
{
	if (strcmp(actual, expected) == 0) {
		return 0;
	}
	fprintf(stderr, "Myocyte reference mismatch for %s: actual=%s expected=%s\n", name, actual, expected);
	return 1;
}

static int reportULLMismatch(const char *name, unsigned long long actual, unsigned long long expected)
{
	if (actual == expected) {
		return 0;
	}
	fprintf(stderr, "Myocyte reference mismatch for %s: actual=%016llx expected=%016llx\n", name, actual, expected);
	return 1;
}

static int reportDecimalULLMismatch(const char *name, unsigned long long actual, unsigned long long expected)
{
	if (actual == expected) {
		return 0;
	}
	fprintf(stderr, "Myocyte reference mismatch for %s: actual=%llu expected=%llu\n", name, actual, expected);
	return 1;
}

static int verifyReference(const char *path, const MyocyteReference *actual)
{
	MyocyteReference expected;
	if (readReference(path, &expected) != 0) {
		rodinia_print_fail("Myocyte reference verification");
		return 1;
	}

	int failed = 0;
	failed |= reportIntMismatch("xmax", actual->xmax, expected.xmax);
	failed |= reportIntMismatch("workload", actual->workload, expected.workload);
	failed |= reportIntMismatch("mode", actual->mode, expected.mode);
	failed |= reportIntMismatch("equations", actual->equations, expected.equations);
	failed |= reportIntMismatch("parameters", actual->parameters, expected.parameters);
	failed |= reportStringMismatch("y_path", actual->y_path, expected.y_path);
	failed |= reportULLMismatch("y_hash", actual->y_hash, expected.y_hash);
	failed |= reportDecimalULLMismatch("y_bytes", actual->y_bytes, expected.y_bytes);
	failed |= reportDecimalULLMismatch("y_lines", actual->y_lines, expected.y_lines);
	failed |= reportStringMismatch("params_path", actual->params_path, expected.params_path);
	failed |= reportULLMismatch("params_hash", actual->params_hash, expected.params_hash);
	failed |= reportDecimalULLMismatch("params_bytes", actual->params_bytes, expected.params_bytes);
	failed |= reportDecimalULLMismatch("params_lines", actual->params_lines, expected.params_lines);
	failed |= reportStringMismatch("output_path", actual->output_path, expected.output_path);
	failed |= reportULLMismatch("output_hash", actual->output_hash, expected.output_hash);
	failed |= reportDecimalULLMismatch("output_bytes", actual->output_bytes, expected.output_bytes);
	failed |= reportDecimalULLMismatch("output_lines", actual->output_lines, expected.output_lines);

	if (failed != 0) {
		rodinia_print_fail("Myocyte reference verification");
		return 1;
	}
	printf("Myocyte reference verification matched '%s'\n", path);
	rodinia_print_pass("Myocyte reference verification");
	return 0;
}
