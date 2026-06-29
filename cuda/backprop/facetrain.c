#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "backprop.h"
#include "backprop_result.h"
#include "omp.h"
#include "../../common/rodinia_verify.h"

enum {
  DEFAULT_RANDOM_SEED = 7
};

int layer_size = 0;
static const float REFERENCE_TOLERANCE = 1.0e-5f;

typedef enum {
  EXECUTION_MODE_RUN = 0,
  EXECUTION_MODE_SAVE_REFERENCE,
  EXECUTION_MODE_VERIFY_REFERENCE
} ExecutionMode;

typedef struct {
  int layer_size;
  ExecutionMode mode;
  const char *reference_path;
} ProgramOptions;

void load(BPNN *net);
void bpnn_train_cuda(BPNN *net, float *eo, float *eh);

static void print_usage(const char *program_name)
{
  fprintf(stderr, "usage: %s <num of input elements>\n", program_name);
  fprintf(stderr, "       %s <num of input elements> --save-reference <path>\n", program_name);
  fprintf(stderr, "       %s <num of input elements> --verify-reference <path>\n", program_name);
}

static int parse_layer_size(const char *text, int *value)
{
  char *end = NULL;
  long parsed_value;

  parsed_value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed_value <= 0) {
    fprintf(stderr, "Invalid input layer size: '%s'\n", text);
    return -1;
  }

  if ((parsed_value % 16) != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    return -1;
  }

  *value = (int) parsed_value;
  return 0;
}

static int parse_mode(const char *flag, ExecutionMode *mode)
{
  if (strcmp(flag, "--save-reference") == 0) {
    *mode = EXECUTION_MODE_SAVE_REFERENCE;
    return 0;
  }

  if (strcmp(flag, "--verify-reference") == 0) {
    *mode = EXECUTION_MODE_VERIFY_REFERENCE;
    return 0;
  }

  fprintf(stderr, "Unknown option: '%s'\n", flag);
  return -1;
}

static int parse_options(int argc, char **argv, ProgramOptions *options)
{
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    return -1;
  }

  if (parse_layer_size(argv[1], &options->layer_size) != 0) {
    return -1;
  }

  options->mode = EXECUTION_MODE_RUN;
  options->reference_path = NULL;

  if (argc == 2) {
    return 0;
  }

  if (parse_mode(argv[2], &options->mode) != 0) {
    print_usage(argv[0]);
    return -1;
  }

  options->reference_path = argv[3];
  return 0;
}

static int handle_reference_mode(
    const ProgramOptions *options,
    const BPNN *net,
    float output_error,
    float hidden_error)
{
  if (options->mode == EXECUTION_MODE_SAVE_REFERENCE) {
    return bpnn_save_reference(net, output_error, hidden_error, options->reference_path);
  }

  if (options->mode == EXECUTION_MODE_VERIFY_REFERENCE) {
    return bpnn_verify_reference(
        net,
        output_error,
        hidden_error,
        options->reference_path,
        REFERENCE_TOLERANCE);
  }

  return 0;
}

static int backprop_face(const ProgramOptions *options)
{
  BPNN *net;
  float out_err, hid_err;
  int status;

  net = bpnn_create(options->layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", options->layer_size);
  load(net);
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  status = handle_reference_mode(options, net, out_err, hid_err);
  bpnn_free(net);
  if (status != 0) {
    if (options->mode == EXECUTION_MODE_VERIFY_REFERENCE) {
      rodinia_print_fail("Backprop reference verification");
    }
    return EXIT_FAILURE;
  }

  printf("Training done\n");
  return EXIT_SUCCESS;
}

int setup(int argc, char **argv)
{
  ProgramOptions options;

  if (parse_options(argc, argv, &options) != 0) {
    return EXIT_FAILURE;
  }

  layer_size = options.layer_size;
  bpnn_initialize(DEFAULT_RANDOM_SEED);

  return backprop_face(&options);
}
