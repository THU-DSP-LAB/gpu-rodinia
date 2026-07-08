#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include <CL/cl.h>
#include <string.h>

int layer_size = 0;
const char *backprop_verify_file = "backprop_result.txt";
int write_validate_file = 0;

static int backprop_face(void)
{
  BPNN *net;
  float out_err, hid_err;
  int status;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  status = bpnn_train_kernel(net, &out_err, &hid_err);
  if (status != 0) {
    bpnn_free(net);
    return status;
  }
  if (write_validate_file) {
    FILE *fp = fopen(backprop_verify_file, "w");
    if (fp == NULL) {
      fprintf(stderr, "backprop: cannot write %s\n", backprop_verify_file);
      bpnn_free(net);
      return -1;
    }
    fprintf(fp, "BACKPROP_OUT_ERR=%0.17f\n", out_err);
    fprintf(fp, "BACKPROP_HID_ERR=%0.17f\n", hid_err);
    fprintf(fp, "BACKPROP_INPUT_HIDDEN_HASH=%llu\n", backprop_last_input_hidden_hash);
    fclose(fp);
  }
  bpnn_free(net);
  printf("\nFinish the training for one iteration\n");
  return 0;
}

int setup(int argc, char **argv)
{
    layer_size = -1;
    const char *verify_output = backprop_verify_file;

    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "--validate") == 0) {
            write_validate_file = 1;
            continue;
        }
        if (strcmp(argv[cur_arg], "--validate-output") == 0 && cur_arg + 1 < argc) {
            verify_output = argv[++cur_arg];
            write_validate_file = 1;
            continue;
        }
        if (strcmp(argv[cur_arg], "-h") == 0) {
            fprintf(stderr, "usage: backprop <-n num of input elements> [-p platform_id] [-d device_id] [-t device_type]\n");
            return 0;
        }
        else if (strcmp(argv[cur_arg], "-n") == 0) {
            if (argc >= cur_arg + 1) {
                layer_size = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-p") == 0) {
            if (argc >= cur_arg + 1) {
                platform_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-d") == 0) {
            if (argc >= cur_arg + 1) {
                device_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
    }

    if (layer_size % 16 != 0){
        fprintf(stderr, "The number of input points must be divided by 16\n");
        return -1;
    }

    int seed = 7;
    bpnn_initialize(seed);
    if (write_validate_file) {
        backprop_verify_file = verify_output;
    }
    return backprop_face();
}
