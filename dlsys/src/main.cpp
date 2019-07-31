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
    // puts("conv2d start");
    float *ACol = new float[n_H * n_W * f * f * n_C];
    float *AnsBatch = ans;
    const float *ABatch = A;
    int DeltaAbatch = n_C_prev * n_W_prev * n_H_prev;
    int DeltaAnsBatch = n_C * n_W * n_H;
    int DeltaACol = f * f * n_C_prev;
    int DeltaX = n_W_prev * n_C_prev;
    for (int i = 0; i < m; i++) {
        float *ASubCol = ACol;
        for (int h = 0; h < n_H; h++) {
            for (int w = 0; w < n_W; w++) {
                int index = 0;
                for (int x = h * stride1; x < h * stride1 + f; x++) {
                    for (int y = w * stride2; y < w * stride2 + f; y++) {
                        for (int k = 0; k < n_C_prev; k++) {
                            ASubCol[index++] = ABatch[
                                    x * DeltaX + y * n_C_prev + k
                            ]; // <=> ABatch[x][y][k]
                        }
                    }
                }
                ASubCol += DeltaACol;
            }
        }
        matmul(ACol, filter, AnsBatch, false, false,
               n_H * n_W, f * f * n_C_prev, n_C);
        ABatch += DeltaAbatch;
        AnsBatch += DeltaAnsBatch;
    }
    delete[] ACol;
    // puts("out conv2d");
}

void Conv2dGradientX(
        float *dX, const float *dH, float *W,
        int m, int n_H_prev, int n_W_prev, int n_C_prev,
        int n_H, int n_W, int n_C,
        int f, int stride1, int stride2
) {
    // puts("conv2d grad x start!");
    float *DXCol = new float[n_H * n_W * f * f * n_C_prev];
    float *DXBatch = dX;
    const float *DHBatch = dH;
    int DeltaDHBatch = n_H * n_W * n_C;
    int DeltaDXBatch = n_H_prev * n_W_prev * n_C_prev;
    int DeltaDXCol = f * f * n_C_prev;
    int DeltaX = f * n_C_prev;
    for (int i = 0; i < m; i++) {
        float *DXSubCol = DXCol;
        matmul(DHBatch, W, DXCol,
               false, true, n_H * n_W, n_C,
               f * f * n_C_prev);
        for (int h = 0; h < n_H; h++) {
            for (int w = 0; w < n_W; w++) {
                int index = 0;
                for (int x = h * stride1; x < h * stride1 + f; x++) {
                    for (int y = w * stride2; y < w * stride2 + f; y++) {
                        for (int k = 0; k < n_C_prev; k++) {
                            DXBatch[x * DeltaX + y * n_C_prev + k]
                                    = DXSubCol[index++];
                        }
                    }
                }
                DXSubCol += DeltaDXCol;
            }
        }
        DHBatch += DeltaDHBatch;
        DXBatch += DeltaDXBatch;
    }
    // puts("conv2d grad x end!");
    delete[] DXCol;
}

void Conv2dGradientW(
        const float *X, const float *dH, float *dW,
        int m, int n_H_prev, int n_W_prev, int n_C_prev,
        int n_H, int n_W, int n_C,
        int f, int stride1, int stride2
) {
    // puts("convgradw start");
    float *DWBatch = new float[f * f * n_C * n_C_prev];
    float *XCol = new float[f * f * n_H * n_W * n_C_prev];
    const float *DHBatch = dH;
    const float *XBatch = X;
    int DeltaDHBatch = n_H * n_C * n_W;
    int DeltaXBatch = n_H_prev * n_W_prev * n_C_prev;
    int DeltaDXCol = f * f * n_C_prev;
    int DeltaX = n_W_prev * n_C_prev;
    for (int i = 0; i < m; i++) {
        float *XSubCol = XCol;
        for (int h = 0; h < n_H; h++) {
            for (int w = 0; w < n_W; w++) {
                int index = 0;
                for (int x = h * stride1; x < h * stride1 + f; x++) {
                    for (int y = w * stride2; y < w * stride2 + f; y++) {
                        for (int k = 0; k < n_C_prev; k++) {
                            XSubCol[index++] = XBatch[
                                    x * DeltaX + y * n_C_prev + k
                            ];
                        }
                    }
                }
                XSubCol += DeltaDXCol;
            }
        }
        matmul(XCol, DHBatch, DWBatch,
               true, false,
               f * f * n_C_prev,
               n_H * n_W, n_C
        );
        for (int r = 0; r < f * f * n_C * n_C_prev; r++) {
            dW[r] += DWBatch[r];
        }
        XBatch += DeltaXBatch;
        DHBatch += DeltaDHBatch;
    }
    delete[] DWBatch;
    delete[] XCol;
    // puts("convgradw end");
}


}
