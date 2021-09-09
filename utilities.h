#include "types.h"

#ifndef UTILITIES_H
#define UTILITIES_H

void hadamardProduct(double *x, double *y, double *res, int length);

double kNearest(double *dist, int *indexValues, int l, int r, int k, int *idx);

int partition(double *dist, int *indexValues, int l, int r);

double kNearestWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right, int k, int *idx);

int partitionWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right);

void swap(double *n1, double *n2);

void swapInts(int *n1, int *n2);

void printResult(knnresult result);

struct knnresult updateKNN(struct knnresult oldResult, struct knnresult newResult );

void dividePoints(int n, int tasks, int * array);

int findDestination(int id, int NumTasks);

int findSender(int id, int NumTasks);

int findBlockArrayIndex(int id, int iteration, int NumTasks);

int findIndexOffset(int id, int iteration, int NumTasks, int * totalPoints);

double findDistance (double * point1, double * point2, int d);

double findMedian(double *distances, int *indexValues, int *offsets, int n, int *idx);

double *mergeArrays(double *arr1, double *arr2, int len1, int len2);

void initializeResult( knnresult *result, int elements, int k);

void insertValueToResult(knnresult *result, double value, int idx, int position, int offset);

#endif //UTILITIES_H
