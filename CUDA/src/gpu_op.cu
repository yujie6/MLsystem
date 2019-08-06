#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
    // Dynamic shared memory, size provided at kernel launch.
    extern __shared__ float loss_per_row[];
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    float maxval = *input_a;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
        loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
    }
    loss_per_row[y] = loss;
    __syncthreads();
    // Compute reduce_mean across rows.
    float mean_loss = 0;
    // Use a single thread to reduce mean across rows.
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int i = 0; i < nrow; ++i) {
            mean_loss += loss_per_row[i];
        }
        mean_loss /= nrow;
        output[0] = mean_loss;
    }
}

__global__ void SoftmaxKernel(int nrow, int ncol, const float *input,
                              float *output) {
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input += y * ncol;
    output += y * ncol;
    float maxval = *input;
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input[x]);
    }
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input[x] - maxval);
    }
    for (int x = 0; x < ncol; ++x) {
        output[x] = exp(input[x] - maxval) / sum;
    }
}

__global__ void ArraySetKernel(float *input, int nrow, int ncol, float value) {
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input += ncol * y;
    for (int i = 0; i < ncol; i++) {
        input[i] = value;
    }
}

__global__ void BroadcastToKernel(const float *input, float *output, int nbroad,
                                  int nrow, int ncol) {
    int id = blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y +
             threadIdx.x;
    //printf("thread is %d\n", id);
    if (id >= nrow * ncol) {
        return;
    }
    int stride = nrow * ncol;
    output += id;
    float value = input[id];
    for (int i = 0; i < nbroad; i++) {
        *output = value;
        output += stride;
    }
}

__global__ void MatrixElementwiseAddKernel(float *input_a, float *input_b,
                                           float *output, int nrow, int ncol) {
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    output += y * ncol;
    for (int i = 0; i < ncol; i++) {
        output[i] = input_a[i] + input_b[i];
    }
}

__global__ void MatrixElementwiseMulKernel(float *input_a, float *input_b,
                                           float *output, int nrow, int ncol) {
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    output += y * ncol;
    for (int i = 0; i < ncol; i++) {
        output[i] = input_a[i] * input_b[i];
    }
}

__global__ void MatrixElementwiseAddByConstKernel(const float *input,
                                                  float *output, int nrow,
                                                  int ncol, float value) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    output[id] = input[id] + value;
}

__global__ void MatrixElementwiseMulByConstKernel(const float *input,
                                                  float *output, int nrow,
                                                  int ncol, float value) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    output[id] = input[id] * value;
}

__global__ void ReluKernel(const float *input, float *output, int nrow,
                           int ncol) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    output[id] = (input[id] > 0) ? input[id] : 0;
}

__global__ void ReluGradientKernel(const float *input, const float *outgrad,
                                   float *output, int nrow, int ncol) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    output[id] = (input[id] > 0) * outgrad[id];
}

__global__ void ReduceSumAxisZeroKernel(const float *input, float *output,
                                        int nreduce, int nrow, int ncol) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    float sum = 0.0;
    int stride = nrow * ncol;
    input += id;
    for (int i = 0; i < nreduce; i++) {
        sum += *input;
        input += stride;
    }
    output[id] = sum;
}

__global__ void MatrixMultiplyKernel(const float *matA, const float *matB,
                                     float *matC, int nrow, int ncol, int len,
                                     bool transposeA, bool transposeB) {
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
             blockDim.x * threadIdx.y + threadIdx.x;
    if (id >= nrow * ncol) {
        return;
    }
    float sum = 0;
    int idy = id / ncol;
    int idx = id % ncol;
    if (!transposeA)
        matA += idy * len;
    else
        matA += idy;
    if (!transposeB)
        matB += idx;
    else
        matB += idx * len;
    for (int i = 0; i < len; i++) {
        sum += (*matA) * (*matB);
        if (!transposeA)
            matA++;
        else
            matA += nrow;
        if (!transposeB)
            matB += ncol;
        else
            matB++;
    }
    matC[id] = sum;
}

__global__ void MatrixMultiplyEXKernel(const float *matA, const float *matB,
                                       float *matC, int nrow, int ncol, int len,
                                       bool transposeA, bool transposeB) {
    int idy = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
              threadIdx.x;
    if (idy >= nrow) {
        return;
    }
    matC += idy * ncol;
    for (int idx = 0; idx < ncol; idx++) {
        float sum = 0;
        const float *matAtmp = matA;
        const float *matBtmp = matB;
        if (!transposeA)
            matAtmp += idy * len;
        else
            matAtmp += idy;
        if (!transposeB)
            matBtmp += idx;
        else
            matBtmp += idx * len;
        for (int j = 0; j < len; j++) {
            sum += (*matAtmp) * (*matBtmp);
            if (!transposeA)
                matAtmp++;
            else
                matAtmp += nrow;
            if (!transposeB)
                matBtmp += ncol;
            else
                matBtmp++;
        }
        *matC = sum;
        matC++;
    }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
    assert(arr->ndim == 2);
    int nrow = arr->shape[0];
    int ncol = arr->shape[1];
    float *input_data = (float *)arr->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    ArraySetKernel<<<1, threads>>>(input_data, nrow, ncol, value);
    return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 3);
    assert(input->shape[0] == output->shape[1] &&
           input->shape[1] == output->shape[2]);
    int nbroad = output->shape[0];
    int nrow = output->shape[1];
    int ncol = output->shape[2];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = numblock;
    BroadcastToKernel<<<blocks, threads>>>(input_data, output_data,
                                                 nbroad, nrow, ncol);

    return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 3);
    assert(output->ndim == 2);
    assert(input->shape[1] == output->shape[0] &&
           input->shape[2] == output->shape[1]);
    int nreduce = input->shape[0];
    int nrow = input->shape[1];
    int ncol = input->shape[2];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = 32;
    blocks.y = (numblock + 32) / 32;
    ReduceSumAxisZeroKernel<<<blocks, threads>>>(input_data, output_data,
                                                 nreduce, nrow, ncol);
    return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(output->ndim == 2);
    assert(matA->shape[0] == matB->shape[0] &&
           matA->shape[1] == matB->shape[1]);
    int nrow = matA->shape[0];
    int ncol = matB->shape[1];
    float *input_data_a = (float *)matA->data;
    float *input_data_b = (float *)matB->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    MatrixElementwiseAddKernel<<<1, threads>>>(input_data_a, input_data_b,
                                               output_data, nrow, ncol);
    return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = 32;
    blocks.y = (numblock + 32) / 32;
    MatrixElementwiseAddByConstKernel<<<blocks, threads>>>(
        input_data, output_data, nrow, ncol, val);
    return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(output->ndim == 2);
    assert(matA->shape[0] == matB->shape[0] &&
           matA->shape[1] == matB->shape[1]);
    int nrow = matA->shape[0];
    int ncol = matB->shape[1];
    float *input_data_a = (float *)matA->data;
    float *input_data_b = (float *)matB->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    MatrixElementwiseMulKernel<<<1, threads>>>(input_data_a, input_data_b,
                                               output_data, nrow, ncol);
    return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = 32;
    blocks.y = (numblock + 32) / 32;
    MatrixElementwiseMulByConstKernel<<<blocks, threads>>>(
        input_data, output_data, nrow, ncol, val);
    return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
    int nrow = matC->shape[0];
    int ncol = matC->shape[1];
    int len;
    if (!transposeA)
        len = matA->shape[1];
    else
        len = matA->shape[0];
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    float *matC_data = (float *)matC->data;
    if (nrow * ncol < 1024 * 1024) {
        dim3 threads(32, 32, 1);
        dim3 blocks;
        int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
        blocks.x = 32;
        blocks.y = (numblock + 32) / 32;
        MatrixMultiplyKernel<<<blocks, threads>>>(matA_data, matB_data,
                                                  matC_data, nrow, ncol, len,
                                                  transposeA, transposeB);
    } else {
        dim3 threads;
        if (nrow <= 1024) {
            threads.x = nrow;
        } else {
            threads.x = 1024;
            threads.y = (nrow + 1023) / 1024;
        }
        MatrixMultiplyEXKernel<<<1, threads>>>(matA_data, matB_data, matC_data,
                                               nrow, ncol, len, transposeA,
                                               transposeB);
    }
    return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = 32;
    blocks.y = (numblock + 32) / 32;
    ReluKernel<<<blocks, threads>>>(input_data, output_data, nrow, ncol);
    return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(in_grad->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    const float *input_data = (const float *)input->data;
    const float *in_grad_data = (const float *)in_grad->data;
    float *output_data = (float *)output->data;
    dim3 threads(32, 32, 1);
    dim3 blocks;
    int numblock = (nrow * ncol + 32 * 32) / (32 * 32);
    blocks.x = 32;
    blocks.y = (numblock + 32) / 32;
    ReluGradientKernel<<<blocks, threads>>>(input_data, in_grad_data,
                                            output_data, nrow, ncol);
    return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->shape[0] == output->shape[0] &&
           input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    assert(nrow <= 1024 * 4);
    int ncol = input->shape[1];
    bool usemoreThreads = false;
    if (!usemoreThreads) {
        const float *input_data = (const float *)input->data;
        float *output_data = (float *)output->data;
        dim3 threads;
        if (nrow <= 1024) {
            threads.x = nrow;
        } else {
            threads.x = 1024;
            threads.y = (nrow + 1023) / 1024;
        }
        SoftmaxKernel<<<1, threads>>>(nrow, ncol, input_data, output_data);
    } else {
        printf("test more threads with shared memory");
    }
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(input_a->shape[0] == input_b->shape[0] &&
           input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float *input_data_a = (const float *)input_a->data;
    const float *input_data_b = (const float *)input_b->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
        nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}
