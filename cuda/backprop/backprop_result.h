#ifndef _BACKPROP_RESULT_H_
#define _BACKPROP_RESULT_H_

#include "backprop.h"

int bpnn_save_reference(
    const BPNN *net,
    float output_error,
    float hidden_error,
    const char *path);

int bpnn_verify_reference(
    const BPNN *net,
    float output_error,
    float hidden_error,
    const char *path,
    float tolerance);

#endif
