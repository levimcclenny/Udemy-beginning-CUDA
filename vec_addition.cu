#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    c[clockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
# define N 512
int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);


    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // cp inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch add kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);

    // cp results to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("%i\n", c);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
