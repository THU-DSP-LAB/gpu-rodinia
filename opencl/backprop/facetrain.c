#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
// #include "omp.h"
#include <CL/cl.h>
#include <string.h>

extern char *strcpy();
extern void exit();

int layer_size = 0;
char ref_file[256] = {0};
char export_file[256] = {0};

void backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("\nFinish the training for one iteration\n");
}

int setup(int argc, char **argv)
{
    layer_size = -1;

    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "-h") == 0) {
            fprintf(stderr, "usage: backprop <-n num of input elements> [-p platform_id] [-d device_id] [-t device_type] [--ref ref_file] [--export export_file]\n");
            exit(0);
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
        else if (strcmp(argv[cur_arg], "--ref") == 0) {
            if (argc >= cur_arg + 1) {
                if(strlen(argv[cur_arg+1]) >= sizeof(ref_file) - 10) {
                    fprintf(stderr, "--ref: file name is too long\n");
                    exit(0);
                }
                strncpy(ref_file, argv[cur_arg+1], sizeof(ref_file) - 1);
                ref_file[sizeof(ref_file) - 1] = '\0';  // Ensure null-termination
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "--export") == 0) {
            if (argc >= cur_arg + 1) {
                if(strlen(argv[cur_arg+1]) >= sizeof(export_file) - 10) {
                    fprintf(stderr, "--export: file name is too long\n");
                    exit(0);
                }
                strncpy(export_file, argv[cur_arg+1], sizeof(export_file) - 1);
                export_file[sizeof(export_file) - 1] = '\0';  // Ensure null-termination
                cur_arg++;
            }
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[cur_arg]);
            exit(0);
        }
    }

    if (layer_size % 16 != 0){
        fprintf(stderr, "The number of input points must be divided by 16\n");
        exit(0);
    }

    int seed = 7;
    bpnn_initialize(seed);
    backprop_face();

    exit(0);
}
