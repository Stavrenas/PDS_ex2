#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include "types.h"

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k);

int MAX_Y_SIZE = 3;

int main(int argc, char *argv[]) {
    struct knnresult result;
    double x[30] = {
            1.0, 2.0,
            0.5, 1.2,
            7.9, 4.6,
            4.0, 0.0,
            8.4, 1.9,
            -1.0, 5.0,
            1.5, 7.2,
            7.1, 3.9,
            -45.0, 10.1,
            28.4, 31.329,
            0.7, 1.4,
            -8.4, -1.9,
            15, 7.2,
            -4.0, 1.1,
            8.4, -31.3};

    double y[30] = {
            1.0, 2.0,
            0.5, 1.2,
            7.9, 4.6,
            4.0, 0.0,
            8.4, 1.9,
            -1.0, 5.0,
            1.5, 7.2,
            7.1, 3.9,
            -45.0, 10.1,
            28.4, 31.329,
            0.7, 1.4,
            -8.4, -1.9,
            15, 7.2,
            -4.0, 1.1,
            8.4, -31.3};

    result = kNN(x, y, 15, 15, 2, 6);

    printResult(result);

    free(result.nidx);
    free(result.ndist);
}

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k) {
    struct knnresult *result = malloc(sizeof(struct knnresult));
    int *indexesRowMajor = (int *) malloc(m * k * sizeof(int));
    double *distRowMajor = (double *) malloc(m * k * sizeof(double));

    double *xx = malloc(n * d * sizeof(double));
    hadamardProduct(x, x, xx, n * d);

    double *xxSum = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        xxSum[i] = cblas_dasum(d, xx + i * d, 1);
    }

    for (int ii = 0; ii < ceil(m / (double) MAX_Y_SIZE); ii++) {
        int partitionSize = MAX_Y_SIZE;
        if (ii * MAX_Y_SIZE + partitionSize > m) {
            partitionSize = m % MAX_Y_SIZE;
        }

        double *currentY = malloc(partitionSize * d * sizeof(double));
        for (int i = 0; i < partitionSize * d; i++) {
            currentY[i] = y[ii * MAX_Y_SIZE * d + i];
        }

        double *yy = malloc(partitionSize * d * sizeof(double));
        hadamardProduct(currentY, currentY, yy, partitionSize * d);

        double *yySum = malloc(partitionSize * sizeof(double));
        for (int i = 0; i < partitionSize; i++) {
            yySum[i] = cblas_dasum(d, yy + i * d, 1);
        }

        double *xy = malloc(n * partitionSize * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, partitionSize, d, -2, x, d, currentY, d, 0, xy, partitionSize);

        double *dist = malloc(n * partitionSize * sizeof(double));
        for (int j = 0; j < partitionSize; j++) {
            for (int i = 0; i < n; i++) {
                double distanceSquared = xxSum[i] + xy[i * partitionSize + j] + yySum[j];
                if (distanceSquared <= 0)
                    dist[j * n + i] = 0;
                else
                    dist[j * n + i] = sqrt(distanceSquared);
            }
        }

        int *indexValues = (int *) malloc(n * partitionSize * sizeof(int));
        for (int i = 0; i < n * partitionSize; i++)
            indexValues[i] = i;

        for (int i = 0; i < partitionSize; i++) {
            for (int j = 0; j < k; j++) {
                int idx;
                distRowMajor[ii * MAX_Y_SIZE * k + i * k + j] = kNearest(dist, indexValues, i * n, (i + 1) * n - 1, j + 1, &idx);
                indexesRowMajor[ii * MAX_Y_SIZE * k + i * k + j] = idx - i * n;
            }
        }

        free(currentY);
        free(yy);
        free(xy);
        free(yySum);
        free(dist);
        free(indexValues);
    }
    free(xx);
    free(xxSum);

    result->ndist = distRowMajor;
    result->nidx = indexesRowMajor;
    result->m = m;
    result->k = k;

    return *result;
}
