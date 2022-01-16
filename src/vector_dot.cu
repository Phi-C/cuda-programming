#include <stdio.h>
/* 这里的N必须是1024的整数倍, 如果不是会出现结果错误, 错误的原因是最后一个block的线程数不是1024, 那么30行的tmp[tidx + k]存在无意义值的情况。
 * 解决这一问题的方法：将向量a和b补0到1024的整数倍。 
 */
#define N 1024 * 64
const int threadsPerBlock =  1024;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

__global__ void vector_dot(float *a, float *b, float *c) {
    __shared__ float tmp[threadsPerBlock];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride  = blockDim.x * gridDim.x;
    int tidx = threadIdx.x;
    float tid_res = 0;
    /*
     * 如果写成:
     * while (tid < N) {
     *     tmp[tidx] += (a[tid] * b[tid]);
     *     tid += stride;
     * } 
     * 会出现计算结果错误, 需要在while语句前加上tmp[tidx] = 0; 做初始化。
     */
    while( tid < N ) {
        tid_res += (a[tid] * b[tid]);
        tid += stride;
    }
    tmp[tidx] = tid_res;
    __syncthreads();
    int k = blockDim.x / 2;
    while(k != 0) {
        if (tidx < k) 
            tmp[tidx] += tmp[tidx + k];
        __syncthreads();
        k = k / 2;
    }

    if (tidx == 0)
        c[blockIdx.x] = tmp[0];
}


int main() {
    size_t size_full = sizeof(float) * N;
    size_t size_partial = sizeof(float) * blocksPerGrid;
    float *a, *b, *c;
    a = (float*)malloc(size_full);
    b = (float*)malloc(size_full);
    c = (float*)malloc(size_partial);
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, size_full);
    cudaMalloc((void**)&dev_b, size_full);
    cudaMalloc((void**)&dev_c, size_partial);
    float host_res = 0;
    for(int i = 0; i < N; i++) {
        a[i] = 2.0 * rand() / RAND_MAX;
        b[i] = 2.0 * rand() / RAND_MAX;
        host_res += (a[i] * b[i]);
    }
    printf("Host Result is %f\n", host_res); 
    cudaMemcpy(dev_a, a, size_full, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size_full, cudaMemcpyHostToDevice);


    float device_res = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    vector_dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, size_partial , cudaMemcpyDeviceToHost);
    for(int i = 0; i < blocksPerGrid; i++)
        device_res += c[i];
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    printf("Device Result is %f\n", device_res);


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}