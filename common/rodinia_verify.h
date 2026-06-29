#ifndef RODINIA_VERIFY_H
#define RODINIA_VERIFY_H

#include <stdio.h>

#define RODINIA_PASS_COLOR "\033[32m"
#define RODINIA_FAIL_COLOR "\033[31m"
#define RODINIA_RESET_COLOR "\033[0m"

static inline int rodinia_print_pass(const char *label)
{
    printf(RODINIA_PASS_COLOR "PASS" RODINIA_RESET_COLOR " %s\n", label);
    return 0;
}

static inline int rodinia_print_fail(const char *label)
{
    printf(RODINIA_FAIL_COLOR "FAIL" RODINIA_RESET_COLOR " %s\n", label);
    return -1;
}

static inline int rodinia_report_check(const char *label, int status)
{
    if (status == 0) {
        return rodinia_print_pass(label);
    }
    return rodinia_print_fail(label);
}

#endif
