#include <stdlib.h>
#include <stdio.h>
#define N 512

__global__ void add(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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

	add<<<1,N>>>(d_a, d_b, d_c);
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


