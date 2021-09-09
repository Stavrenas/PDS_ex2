#include <time.h> // clock_gettime, timespec
#include <stdint.h>
#include "types.h"

double measureTimeForRunnable(knnresult (*runnable)(double *x, int n, int d, int k), double *arg1, int arg2, int arg3, int arg4, knnresult *result) {
    struct timespec ts_start;
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    *result = runnable(arg1, arg2, arg3, arg4);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    return (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000000.0;
}
