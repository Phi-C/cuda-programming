#include <cstdio>
#include <iostream>
#include <algorithm>
// 实现一个C=AB的矩阵乘法, 其中A=[1024, 64], B=[64, 1024]
const int featBlocks = 32;
const int TileDim = 32;
const int M = 1024 * 1;
const int K = TileDim * featBlocks;
const int N = 1024 * 1;
// maxThreadsPerBlock = 1024, 所以threadsPerBlock.x * threadsPerBlock.y <= 1024
// 对任何一台机器, 要先参考hello-world.cu代码, 了解一个block最多包含多少个线程
dim3 threadsPerBlock(32, 32);
dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


__global__ void matrix_mul_simple(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0.0f;
    for(int i = 0; i < K; i++)
        res += (a[row * K + i] * b[col + i * N]);
    c[row * N + col] = res;
}

__global__ void matrix_mul_optimized_v1(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float ATile[TileDim][TileDim];
    float res = 0;
    for(int featBlockIdx = 0; featBlockIdx < featBlocks; featBlockIdx++) {
        // 这里对global memory的读取每次读取32*4=128字节, 然后跨越N *4 个字节继续读取连续128字节; 由于每次都读取128字节, 属于合并访问
        ATile[threadIdx.y][threadIdx.x] = a[row * K  + featBlockIdx * TileDim + threadIdx.x];
        // 写ATile和读ATile的线程都是同一个warp的, 所以只需要__syncwarp()就足够了
        __syncwarp();
        for(int i = 0; i < TileDim; i++) {
            res += (ATile[threadIdx.y][i] * b[col + (featBlockIdx * TileDim + i) * N]);
        }
    }
    c[row * N + col] = res;
}

__global__ void matrix_mul_optimized_v2(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float ATile[TileDim][TileDim];
    __shared__ float BTile[TileDim][TileDim];
    float res = 0.0f;
    for(int featBlockIdx = 0; featBlockIdx < featBlocks; featBlockIdx++) {
        // 这里对global memory的读取每次读取32*4=128字节
        ATile[threadIdx.y][threadIdx.x] = a[row * K  + featBlockIdx * TileDim + threadIdx.x];
        BTile[threadIdx.y][threadIdx.x] = b[col + (featBlockIdx * TileDim + threadIdx.y) * N];
        // 写BTile和读BTile的线程不是同一个warp的, 但是同一个block的
        __syncthreads();
        for(int i = 0; i < TileDim; i++) {
            res += (ATile[threadIdx.y][i] * BTile[i][threadIdx.x]);
        }
        __syncthreads();
    }
    c[row * N + col] = res;
}

bool ValResult(float *base, float *test) {
    // 验证计算结果
    for(int i = 0; i < M*N; i++)
        if (abs(base[i] - test[i]) > 0.1) {
            std::cout << i << ": " <<  base[i] << " VS " << test[i] << std::endl;
            return false;
        }
    return true;
}

float GetKernelExecTime(dim3 grids, dim3 blocks, float *dev_a, float *dev_b, float *dev_c, void(*kernel)(float *a, float *b, float *c)) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);    
    kernel<<<grids, blocks>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

int main() {
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;
    float *host_a, *host_b, * host_c, *dev_a, *dev_b, *dev_c;
    host_a = (float*)malloc(size_a);
    host_b = (float*)malloc(size_b);
    host_c = (float*)malloc(size_c);
    cudaMalloc((void**)&dev_a, size_a);
    cudaMalloc((void**)&dev_b, size_b);
    cudaMalloc((void**)&dev_c, size_c);
    for(int i = 0; i < M * K; i++)
        host_a[i] = 5.0 * rand() / RAND_MAX;
    for(int i = 0; i < K * N; i++)
        host_b[i] = 5.0 * rand() / RAND_MAX;

    // 使用clock_t计算时间
    clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    float *baseline;
    baseline = (float*)malloc(size_c);
    for(int row = 0; row < M; row++) {
        for(int col = 0; col < N; col++) {
            float sum = 0;
            for(int idx = 0; idx < K; idx++)
                sum += host_a[row * K + idx] * host_b[col + idx * N];
            baseline[row * N + col] = sum;
        }
    }
    stop_cpu = clock();
    printf("It took %f seconds to do matrix multiplicaiton in CPU\n", (stop_cpu - start_cpu) / (float)CLOCKS_PER_SEC);

    cudaMemcpy(dev_a, host_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size_b, cudaMemcpyHostToDevice);

    float time1 = GetKernelExecTime(blocksPerGrid, threadsPerBlock, dev_a, dev_b, dev_c, matrix_mul_simple);
    printf("It took %f seconds to do multrix multiplicaiton in GPU\n", time1);
    cudaMemcpy(host_c, dev_c, size_c, cudaMemcpyDeviceToHost);
    std::cout << "ValResult for matrix_mul_simple is: " << std::boolalpha << ValResult(baseline, host_c) << std::endl;

    float time2 = GetKernelExecTime(blocksPerGrid, threadsPerBlock, dev_a, dev_b, dev_c, matrix_mul_optimized_v1);
    printf("It took %f seconds to do multrix multiplicaiton optimized version 1 in GPU\n", time2);
    cudaMemcpy(host_c, dev_c, size_c, cudaMemcpyDeviceToHost);
    std::cout << "ValResult for matrix_mul_simple_optimized_v1 is: " << std::boolalpha << ValResult(baseline, host_c) << std::endl;

    float time3 = GetKernelExecTime(blocksPerGrid, threadsPerBlock, dev_a, dev_b, dev_c, matrix_mul_optimized_v2);
    printf("It took %f seconds to do multrix multiplicaiton optimized version 2 in GPU\n", time3);
    cudaMemcpy(host_c, dev_c, size_c, cudaMemcpyDeviceToHost);
    std::cout << "ValResult for matrix_mul_simple_optimized_v2 is: " << std::boolalpha << ValResult(baseline, host_c) << std::endl;      

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    free(baseline);

    return 0;
}