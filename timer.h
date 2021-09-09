#include "types.h"

#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>

double measureTimeForRunnable(knnresult (*runnable)(double *x, int n, int d, int k), double *arg1, int arg2, int arg3, int arg4, knnresult *result);

#endif //T1_TIMER_H
