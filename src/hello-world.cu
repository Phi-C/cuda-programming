#include <iostream>
#include <stdio.h>
#include <vector>

int main() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount( &count);
    std::cout << "There are " << count << " devices" << std::endl;
    cudaGetDeviceProperties( &prop, 0);
    // 查看是否允许device overlap
    if (prop.deviceOverlap)
        // RTX 3060是允许deviceOverlap的, 这样可以定义多个CUDA stream, 使得数据传输和运算能够同时进行
        std::cout << "Enalbe device overlap" << std::endl;
    else
        std::cout << "Disable device overlap" << std::endl;
    std::cout << "---------------Memory Information--------------" << std::endl;
    // RTX 3060的global memory是6GB, 和nvidia-smi上面显示的一致
    printf("Total global mem: %ld Bytes (%f GB)\n", prop.totalGlobalMem, prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 );
    printf("Total constant mem: %ld Bytes (%f KB)\n", prop.totalConstMem, prop.totalConstMem / 1024.0 );
    printf("Number of SM: %d\n", prop.multiProcessorCount );
    printf("Shared mem per block: %ld Bytes (%f KB)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0 );
    printf("Resiters per block: %d\n", prop.regsPerBlock );
    printf("Threads in warp: %d\n", prop.warpSize );
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock );
    printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

    std::cout << "Hello" << std::endl;
    return 0;
}
