# Assignment Bonus: GPU CUDA Test

[PPCA 2019](https://acm.sjtu.edu.cn/wiki/PPCA_2019) machine learning system Bonus - *CUDA*

In this assignment, we would implement some GPU kernel for ML System.

Key concepts and data structures that we would need to implement are
- GPU kernel implementations of common kernels, e.g. Relu, MatMul, Softmax.

## Overview of Module
- tests/dlsys/autodiff.py: Implements computation graph, autodiff, GPU/Numpy Executor.
- tests/dlsys/gpu_op.py: Exposes Python function to call GPU kernels via ctypes.
- tests/dlsys/ndarray.py: Exposes Python GPU array API.

- src/dlarray.h: header for GPU array.
- src/c_runtime_api.h: C API header for GPU array and GPU kernels.
- src/gpu_op.cu: cuda implementation of kernels 

## What you need to do?
Understand the code skeleton and tests. Fill in implementation wherever marked `"""TODO: Your code here"""`.

There are only one file with TODOs for you.
- src/gpu_op.cu

### Special note
Do not change Makefile to use cuDNN for GPU kernels. Also, cublas is forbidden for matrix multiply.

## Environment setup
- You need to install CUDA toolkit ([instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/)) on your own machine, and set the environment variables.
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  export PATH=/usr/local/cuda/bin:$PATH
  ```
- Workstation in the lab is equipped with CUDA, you can use it directly.
- MacBook (Pro) is not equipped with NVIDIA GPU, so mac users need coding with the WorkStation in the lab.

## Tests cases
We have 12 tests in tests/test_gpu_op.py. We would grade your GPU kernel implementations based on those tests.

Compile
```bash
make
```

Run all tests with
```bash
# sudo pip install nose
nosetests -v tests/test_gpu_op.py
```

If your implementation is correct, you would see

Profile GPU execution with
```bash
nvprof nosetests -v tests/test_gpu_op.py
```

### Grading rubrics
- test_gpu_op.test_array_set ... 0.5 pt
- test_gpu_op.test_broadcast_to ... 0.5 pt
- test_gpu_op.test_reduce_sum_axis_zero ... 1 pt
- test_gpu_op.test_matrix_elementwise_add ... 0.5 pt
- test_gpu_op.test_matrix_elementwise_add_by_const ... 0.5 pt
- test_gpu_op.test_matrix_elementwise_multiply ... 0.5 pt
- test_gpu_op.test_matrix_elementwise_multiply_by_const ... 0.5 pt
- test_gpu_op.test_matrix_multiply ... 3 pt
- test_gpu_op.test_relu ... 1 pt
- test_gpu_op.test_relu_gradient ... 1 pt
- test_gpu_op.test_softmax ... 1 pt
- test_gpu_op.test_softmax_cross_entropy ... Implemented.
