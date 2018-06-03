#include <stdio.h>
#include <stdlib.h>

__global__
void cudaMultVectorsKernel(int N, float *x, float *y, float *z)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) {
    z[idx] = x[idx] * y[idx];
  }
  // idx = idx + blockDim.x * gridDim.x; // we will discuss this later...
}

// extern "C" is necessary because nvcc uses c++ compiler to compile cuda code
// hence applies name mangling. Because we use gcc for linking, we should 
// prevent name mangling.
extern "C"
void runKernel(int N, float *x, float *y, float *z) {
    cudaMultVectorsKernel<<<(N+511)/512, 512>>>(N, x, y, z);
}
