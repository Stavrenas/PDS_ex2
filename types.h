
#ifndef TYPES_H
#define TYPES_H

typedef struct vpNode {
    struct vpNode *parent;
    double *vp;
    int vpIdx;
    double mu;
    struct vpNode *left;
    struct vpNode *right;
} vpNode;

typedef struct knnresult {
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
} knnresult;

#endif //TYPES_H
