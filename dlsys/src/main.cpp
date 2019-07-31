#include <iostream>
#include <cstdio>
//#include <omp.h>
#include <cblas.h>

using namespace std;

extern "C" {

void matmul(const float *A, const float *B, float *C,
            bool TA, bool TB, int m, int k, int n) {
    //puts("in matmul");
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE trans_A = TA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trans_B = TB ? CblasTrans : CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;
    const int lda = TA ? m : k;
    const int ldb = TB ? k : n;
    const int ldc = n;
    cblas_sgemm(order, trans_A, trans_B,
                m, n, k, alpha,
                A, lda,
                B, ldb, beta,
                C, ldc
    );
}

void conv2d(const float *A, const float *filter, float *ans,
            int m, int n_H, int n_W, int n_C,
            int n_H_prev, int n_W_prev, int n_C_prev,
            int f, int stride1, int stride2
) {
    float * A_sub_col = new float[n_H * n_W, f * f];
    for (int i = 0; i < m; i++) {
        return;
    }
}

}
