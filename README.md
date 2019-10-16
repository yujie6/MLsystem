# Naive ML System
## Project Structure  
### 1. `/FashMnist`
Use `tensorflow` to build a network for FashionMnist classification problem with
cnn and linear regression. 
#### Reference 
1. <a href="https://www.coursera.org/learn/machine-learning/">Andrew Ng's ML course</a>
2. <a href="https://www.coursera.org/learn/convolutional-neural-networks">CNN course on coursera</a>
### 2. `/autodiff` 
Build computation graph to implement auto gradient, also a good
preparation for the next project.
#### Reference  
1. <a href="http://dlsys.cs.washington.edu/pdf/lecture4.pdf">University of 
Washington course project</a>
2. <a href="https://blog.csdn.net/aws3217150/article/details/70214422">Introduction 
to autodiff</a>

### 3. `/Tinyflow`
Build a python package named `tinyflow` with the same API as tensorflow. And use 
clib to boost the speed.  

### 4. `/CUDA` 
Use `nvcc` to implement some operations in tensorflow, and 
execute with GPU.
Environment setup for linux:
```bash
sudo apt install nvidia-cuda-toolkit
```



