#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include <float.h>
#include "utilities.h"
#include "types.h"

void hadamardProduct(double *x, double *y, double *res, int length) {
    for (int i = 0; i < length; i++)
        res[i] = x[i] * y[i];
}

// k is one based
double kNearest(double *dist, int *indexValues, int left, int right, int k, int *idx) {
    int pivot = partition(dist, indexValues, left, right);

    if (k < pivot - left + 1) {
        return kNearest(dist, indexValues, left, pivot - 1, k, idx);
    } else if (k > pivot - left + 1) {
        return kNearest(dist, indexValues, pivot + 1, right, k - pivot + left - 1, idx);
    } else {
        *idx = indexValues[pivot];
        return dist[pivot];
    }
}

int partition(double *dist, int *indexValues, int left, int right) {
    double x = dist[right];
    int i = left;
    for (int j = left; j <= right - 1; j++) {
        if (dist[j] <= x) {
            swap(&dist[i], &dist[j]);
            swapInts(&indexValues[i], &indexValues[j]);
            i++;
        }
    }
    swap(&dist[i], &dist[right]);
    swapInts(&indexValues[i], &indexValues[right]);
    return i;
}

// k is one based
double kNearestWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right, int k, int *idx) {
    int pivot = partitionWithOffsets(dist, indexValues, offsets, left, right);

    if (k < pivot - left + 1) {
        return kNearestWithOffsets(dist, indexValues, offsets, left, pivot - 1, k, idx);
    } else if (k > pivot - left + 1) {
        return kNearestWithOffsets(dist, indexValues, offsets, pivot + 1, right, k - pivot + left - 1, idx);
    } else {
        *idx = indexValues[pivot];
        return dist[pivot];
    }
}

int partitionWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right) {
    double x = dist[right];
    int i = left;
    for (int j = left; j <= right - 1; j++) {
        if (dist[j] <= x) {
            swap(&dist[i], &dist[j]);
            swapInts(&indexValues[i], &indexValues[j]);
            swapInts(&offsets[i], &offsets[j]);
            i++;
        }
    }
    swap(&dist[i], &dist[right]);
    swapInts(&indexValues[i], &indexValues[right]);
    swapInts(&offsets[i], &offsets[right]);
    return i;
}

void swap(double *n1, double *n2) {
    double temp = *n1;
    *n1 = *n2;
    *n2 = temp;
}

void swapInts(int *n1, int *n2) {
    int temp = *n1;
    *n1 = *n2;
    *n2 = temp;
}

void printResult(knnresult result) {
    for (int i = 0; i < result.m * result.k; i++) {
        if (i % result.k == 0)
            printf("\n");
        printf("%16.4f(%08d) ", result.ndist[i], result.nidx[i]);
    }
    printf("\n");
}

void dividePoints(int n, int tasks, int *array) {
    int points = n / tasks;
    for (int i = 0; i < n; i++) {
        array[i] = points;
    }

    int pointsLeft = n % tasks;
    for (int i = 0; pointsLeft > 0; i++) {
        array[i]++;
        pointsLeft--;
    }
}

int findDestination(int id, int NumTasks) {
    if (id == NumTasks - 1)
        return 0;
    else
        return (id + 1);
}

int findSender(int id, int NumTasks) {
    if (id == 0)
        return (NumTasks - 1);
    else
        return (id - 1);
}

struct knnresult updateKNN(struct knnresult oldResult, struct knnresult newResult) {
    struct knnresult *result = malloc(sizeof(struct knnresult));

    int k, kmin;
    int m = oldResult.m; //==newResult.m
    if (oldResult.k > newResult.k) {
        k = oldResult.k;
        kmin = newResult.k;
    } else {
        k = newResult.k;
        kmin = oldResult.k;
    }

    double *newNearest = malloc(m * k * sizeof(double));
    int *newIndexes = malloc(m * k * sizeof(int));

    for (int i = 0; i < m; i++) {
        int it1, it2; //iterator for old and new result.ndist
        it1 = it2 = i * k;
        for (int j = 0; j < k; j++) {
            int flag = 0;
            do {
                flag = 0;
                for (int jj = i * k; jj < i * k + j; jj++) {
                    // Check if indexes at it1 and it2 have been included already
                    if (newIndexes[jj] == newResult.nidx[it1]) {
                        flag = 1;
                        it1++;
                    }
                    if (newIndexes[jj] == oldResult.nidx[it2]) {
                        flag = 1;
                        it2++;
                    }
                }
            } while (flag == 1);

            if (newResult.ndist[it1] <= oldResult.ndist[it2] && it1 < (i + 1) * k) {
                newNearest[i * k + j] = newResult.ndist[it1];
                newIndexes[i * k + j] = newResult.nidx[it1];
                if (newResult.nidx[it1] == oldResult.nidx[it2])
                    it2++;
                it1++;
            } else if (newResult.ndist[it1] > oldResult.ndist[it2] && it2 < (i + 1) * k) {
                newNearest[i * k + j] = oldResult.ndist[it2];
                newIndexes[i * k + j] = oldResult.nidx[it2];
                if (newResult.nidx[it1] == oldResult.nidx[it2])
                    it1++;
                it2++;
            } else {
                newNearest[i * k + j] = INFINITY;
                newIndexes[i * k + j] = -1;
            }
        }
    }

    result->ndist = newNearest;
    result->nidx = newIndexes;
    result->m = oldResult.m;
    result->k = oldResult.k;
    return *result;

}

int findBlockArrayIndex(int id, int iteration, int NumTasks) { //iteration >=1
    int Y = id - iteration;
    if (Y < 0)
        Y += NumTasks;
    return Y;
}

int findIndexOffset(int id, int iteration, int NumTasks, int *totalPoints) { //total points is the number of the points before
    int Y = findBlockArrayIndex(id, iteration, NumTasks);
    int result = 0;
    for (int i = 0; i < Y; i++)
        result += totalPoints[i];

    return result;
}


double findDistance(double *point1, double *point2, int d) {
    double distance = 0;
    for (int i = 0; i < d; i++)
        distance += pow((point1[i] - point2[i]), 2);
    return sqrt(distance);
}

double findMedian(double *distances, int *indexValues, int *offsets, int n, int *idx) {
    if (n % 2 == 0) {
        int idx2;
        return (kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2, idx) +
                kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2 + 1, &idx2)) / 2;
    } else {
        return kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2 + 1, idx);
    }
}

void insertValueToResult(knnresult *result, double value, int idx, int position, int offset) {
    for (int j = offset + result->k - 1; j > position; j--) {
        result->ndist[j] = result->ndist[j - 1];
        result->nidx[j] = result->nidx[j - 1];
    }
    result->ndist[position] = value;
    result->nidx[position] = idx;
}

double *mergeArrays(double *arr1, double *arr2, int len1, int len2) {
    double *merged = (double *) malloc((len1 + len2) * sizeof(double));
    for (int i = 0; i < len1; i++)
        merged[i] = arr1[i];
    for (int i = 0; i < len2; i++)
        merged[len1 + i] = arr2[i];
    return merged;
}

void initializeResult(knnresult *result, int elements, int k) {
    result->ndist = (double *) malloc(elements * k * sizeof(double));
    result->nidx = (int *) malloc(elements * k * sizeof(int));
    result->k = k;
    result->m = elements;
    for (int ii = 0; ii < result->m * k; ii++) {
        result->ndist[ii] = INFINITY;
        result->nidx[ii] = -1;
    }
}
