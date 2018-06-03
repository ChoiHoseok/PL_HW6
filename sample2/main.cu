#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

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

int main() {
    const int N = 1<<20;
    float *x, *y, *z, *dx, *dy, *dz;
	
	//printf("hello world\n");
    cudaMalloc((void**)&dx, N*sizeof(float));
    cudaMalloc((void**)&dy, N*sizeof(float));
    cudaMalloc((void**)&dz, N*sizeof(float));

    // init array x, y
    for (int i=0; i<N; i++) {
        x[i] = 2.3f*i;
        y[i] = 4.1f*i;
    }

    cudaMemcpy(dx, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N*sizeof(float), cudaMemcpyHostToDevice);

    runKernel(N, dx, dy, dz);

    cudaMemcpy(z, dz, N*sizeof(float), cudaMemcpyDeviceToHost);
	//printf("hello world\n");  
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
	//printf("hello world\n");
    free(x);
    free(y);
    free(z);
	//printf("hello world\n");
    return 0;
}


