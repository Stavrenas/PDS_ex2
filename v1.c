#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include <mpi.h>
#include <string.h>
#include "utilities.h"
#include "types.h"
#include "controller.h"
#include "read.h"

knnresult distrAllkNN(double *x, int n, int d, int k);

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset);

int main(int argc, char *argv[]) {
    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);
    
    int n;
    int d;
    int k=atoi(argv[2]);

    double * X=read_X(&n,&d,argv[1]);

    char *resFilename = (char *) malloc((40 + strlen(argv[1])) * sizeof(char));
    sprintf(resFilename, "v1_res_%s_%07d_%04d_%04d_%04d.txt", argv[1], n, d, k, NumTasks);
    knnresult mergedResult = runAndPresentResult(distrAllkNN, X, n, d, k, argv[1], "v1", resFilename);
    free(resFilename);

    if (SelfTID == 0) {    //send every result to the first process for printing
        char *outFilename = (char *) malloc((40 + strlen(argv[1])) * sizeof(char));
        sprintf(outFilename, "v1_out_%s_%07d_%04d_%04d_%04d.txt", argv[1], n, d, k, NumTasks);
        FILE *f = fopen(outFilename, "wb");
        for (int i = 0; i < mergedResult.m * mergedResult.k; i++) {
            if (i % mergedResult.k == 0)
                fprintf(f, "\n");
            fprintf(f, "%16.4f(%08d) ", mergedResult.ndist[i], mergedResult.nidx[i]);
        }
        for (int i = 1; i < NumTasks; i++) {
            double *incomingDistances = malloc(mergedResult.m * mergedResult.k * sizeof(double));
            int *incomingIndexes = malloc(mergedResult.m * mergedResult.k * sizeof(int));
            MPI_Recv(incomingDistances, mergedResult.m * mergedResult.k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &mpistat);
            MPI_Recv(incomingIndexes, mergedResult.m * mergedResult.k, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &mpistat);
            int incomingLength;
            MPI_Get_count(&mpistat, MPI_INTEGER, &incomingLength);
            for (int i = 0; i < incomingLength; i++) {
                if (i % mergedResult.k == 0)
                    fprintf(f, "\n");
                fprintf(f, "%16.4f(%08d) ", incomingDistances[i], incomingIndexes[i]);
            }
            free(incomingDistances);
            free(incomingIndexes);
        }
        fprintf(f, "\n");
        fclose(f);
        free(outFilename);
    } else {
        MPI_Send(mergedResult.ndist, mergedResult.m * mergedResult.k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Isend(mergedResult.nidx, mergedResult.m * mergedResult.k, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &mpireq);
    }

    MPI_Finalize();
    return 0;
}

knnresult distrAllkNN(double *x, int n, int d, int k) {

    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    int *totalPoints = (int *) malloc(n * sizeof(int));
    dividePoints(n, NumTasks, totalPoints);
    int points = totalPoints[SelfTID];
    int offset = 0;

    double *X = (double *) malloc(points * d * sizeof(double));
    for (int i = 0; i < SelfTID; i++) {
        offset += totalPoints[i];
    }
    X = x + offset * d;

    int sentElements = totalPoints[SelfTID] * d;
    MPI_Isend(X, sentElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq);  //send self block

    int sender = findSender(SelfTID, NumTasks);
    int receivedArrayIndex, receivedElements; //initialize variables for use in loop
    int loopOffset;

    struct knnresult result, previousResult, newResult, mergedResult;
    result = smallKNN(X, X, points, points, d, k, offset);
    previousResult = result;
    mergedResult = result;

    for (int i = 1; i < NumTasks; i++) {  //ring communication

        receivedArrayIndex = findBlockArrayIndex(SelfTID, i, NumTasks);
        receivedElements = totalPoints[receivedArrayIndex] * d;
        double *Y = malloc(receivedElements * sizeof(double));

        MPI_Recv(Y, receivedElements, MPI_DOUBLE, sender, 55, MPI_COMM_WORLD, &mpistat); //receive from previous process
        MPI_Isend(Y, receivedElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq); //send the received array to next process

        loopOffset = findIndexOffset(SelfTID, i, NumTasks, totalPoints);
        newResult = smallKNN(X, Y, points, totalPoints[receivedArrayIndex], d, k, loopOffset);
        mergedResult = updateKNN(newResult, previousResult);
        previousResult = mergedResult;

    }

    return mergedResult;
}

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset) {
    struct knnresult *result = malloc(sizeof(struct knnresult));

    double *xx = malloc(n * d * sizeof(double));
    hadamardProduct(x, x, xx, n * d);

    double *yy = malloc(m * d * sizeof(double));
    hadamardProduct(y, y, yy, m * d);

    double *xxSum = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        xxSum[i] = cblas_dasum(d, xx + i * d, 1);
    }

    double *yySum = malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        yySum[i] = cblas_dasum(d, yy + i * d, 1);
    }

    double *xy = malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, xy, m);

    double *dist = malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double distanceSquared = xxSum[i] + xy[i * m + j] + yySum[j];
            if (distanceSquared <= 0)
                dist[i * m + j] = 0;
            else
                dist[i * m + j] = sqrt(distanceSquared);
        }
    }

    int *indexValues = (int *) malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++)
        indexValues[i] = i + indexOffset;

    int kMin = k > n ? n : k;
    kMin = kMin > m ? m : kMin;

    int *indexesRowMajor = (int *) malloc(n * k * sizeof(int));
    double *distRowMajor = (double *) malloc(n * k * sizeof(double));
    for (int i = 0; i < n * k; i++) {
        distRowMajor[i] = INFINITY;
        indexesRowMajor[i] = -1;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < kMin; j++) {
            int idx;
            distRowMajor[i * k + j] = kNearest(dist, indexValues, i * m, (i + 1) * m - 1, j + 1, &idx);
            indexesRowMajor[i * k + j] = idx - i * m;
        }
    }

    result->ndist = distRowMajor;
    result->nidx = indexesRowMajor;
    result->m = n;
    result->k = k;

    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    return *result;
}
