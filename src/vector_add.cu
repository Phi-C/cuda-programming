#include <stdio.h>

#define N 100000000

__global__ void vector_add(float *dev_a, float *dev_b, float *dev_c) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;
    while( tid < N) {
        dev_c[tid] = dev_a[tid] + dev_b[tid];
        tid += stride;
    }
}

int main() {
    // allocate host memory and prepare data on host
    size_t  size = N * sizeof(float);
    /* 如果通过float a[N], b[N], c[N];的方式在设备上分配内存, 当N=100w是会出现Segmentation Fault
        通过malloc()动态分配则不会.
       静态数组a[N]存放在stack中, 如果通过malloc()分配的动态数组存放在hea p中. 栈内存和堆内存相比很小.
       通过ulimit -s可以查看linux系统下的栈空间大小,我的是8192 KB
       100W * 4 Bytes * 3 = 1200W Bytes ≈ 1W KB
     */
    float *a, *b, *c;
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    for(long i = 0; i < N; i++) {
        a[i] = rand() / RAND_MAX;
        b[i] = rand() / RAND_MAX;
    }

    // allocate device memory
    /* 规范性：*紧跟变量名，不推荐float* dev_a写法 */
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // copy data from host to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // lauch the kernel and transfer data back to host
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    long blocks = 1024;
    long grids = (N + blocks - 1) / blocks;
    printf("blockNumPerGrid: %ld\nthreadNumPerBlock: %ld\n", grids, blocks);
    cudaEventRecord(start, 0);
    vector_add<<<grids, blocks>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Computate %d elements vector addition took %f seconds.\n", N, time);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // verificate the result
    for(long i = 0; i < N; i++)
        if(a[i]+b[i] != c[i]) {
            printf("Get the wrong result");
            break;
        }

    // free resources
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    
    return 0;
}