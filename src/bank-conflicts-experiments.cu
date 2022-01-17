#include <stdio.h>

const int N = 1024000;
const int threadsPerBlock = 1024;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void fetch_two_way_conflicts(float *a) {
    __shared__ float BlockCache[threadsPerBlock * 2];
    /* 这样处理相当于将位于全局内存的向量a的N个元素划分成gridDim.x个区块, 每个block内的shared memory缓存自己对应的区块和下一个区块 */
    int tidInBlock = threadIdx.x;
    int baseIdx = blockIdx.x * blockDim.x;
    BlockCache[2*tidInBlock] = a[baseIdx + 2*tidInBlock];
    BlockCache[2*tidInBlock + 1] = a[baseIdx + 2*tidInBlock + 1];
    __syncthreads();  
}

__global__ void fetch_no_conflicts(float *a) {
    __shared__ float BlockCache[threadsPerBlock * 2];
    /* 这样处理相当于将位于全局内存的向量a的N个元素划分成gridDim.x个区块, 每个block内的shared memory缓存自己对应的区块和下一个区块 */
    int tidInBlock = threadIdx.x;
    int baseIdx = blockIdx.x * blockDim.x;
    BlockCache[tidInBlock] = a[baseIdx + tidInBlock];
    BlockCache[tidInBlock + blockDim.x] = a[baseIdx + tidInBlock + blockDim.x];
    __syncthreads();  
}

int main() {
    size_t size = sizeof(float) * N;
    float *host_a;
    host_a = (float*)malloc(size);
    float *dev_a;
    cudaMalloc((void**)&dev_a, size);

    for(int i = 0; i < N; i++) {
        host_a[i] = rand() / RAND_MAX;
    }
    cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);

    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);
    fetch_two_way_conflicts<<<blocksPerGrid, threadsPerBlock>>>(dev_a);
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float time1;
    cudaEventElapsedTime(&time1, start1, stop1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    printf("It took %f seconds to fetch data from global memory to shared memory with 2-way bank conflicts\n", time1);

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);
    fetch_no_conflicts<<<blocksPerGrid, threadsPerBlock>>>(dev_a);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float time2;
    cudaEventElapsedTime(&time2, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    printf("It took %f seconds to fetch data from global memory to shared memory with no bank conflicts\n", time2);

    cudaFree(dev_a);
    free(host_a);

    return 0;
}