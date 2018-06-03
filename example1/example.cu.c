#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void kernel( int *a, int dimx, int dimy ) {
	int ix = blockIdx.x*blockDim.x + threadIdx.x; 
	int iy = blockIdx.y*blockDim.y + threadIdx.y; 
	int idx = iy*dimx + ix;
	a[idx] = a[idx]+1;
}

int main() {
	int dimx = 16, dimy = 16;
	int num_bytes = dimx*dimy*sizeof(int);
	int *d_a=0, *h_a=0; // device and host pointers h_a = (int*)malloc(num_bytes);
	
	cudaMalloc( (void**)&d_a, num_bytes );
	
	if( 0==h_a || 0==d_a ) {
		printf("couldn't allocate memory\n"); return 1;
	}
	cudaMemset( d_a, 0, num_bytes );
	
	dim3 grid, block;
	block.x = 4; block.y = 4;
	grid.x = dimx / block.x;
	grid.y = dimy / block.y;
	kernel<<<grid, block>>>( d_a, dimx, dimy );
	
	cudaMemcpy(h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
	
	free( h_a ); cudaFree( d_a ); 
	
	return 0;
}