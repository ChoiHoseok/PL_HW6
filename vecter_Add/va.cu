#include <stdlib.h>
#include <stdio.h>
#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n)
		c[index] = a[index] + b[index];
}

void random_ints(int* a);
int main(void){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);
	int i;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);


	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	random_ints(a);
	random_ints(b);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	for(i = 0; i < 512; i++){
		printf("%d ",c[i]);
		if(i%5 == 0)
			printf("\n");
	}
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_a);
	cudaFree(d_a);

	return 0;
}

void random_ints(int* a)
{
	int i;
	for ( i = 0; i < 512; ++i)
		a[i] = rand();
}


