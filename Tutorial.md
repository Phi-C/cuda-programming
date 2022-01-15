# 1 基本概念
## 1.1 硬件常识
SM(Stream Multiprocessor)

SP(Stream Processor): also known as CUDA Cores

要运行CUDA程序，需要安装：1）NVIDIA显卡驱动 （安装好显卡驱动后使用nvidia-smi就可以看到设备信息，上面会显示CUDA的驱动版本号）；2）CUDA Toolkit(用来编译CUDA程序的nvcc就包含在CUDA Toolkt里，CUDA的运行时版本就是CUDA Toolkit的版本号)
> nvcc hello-world.cu -o hello-world.out

nvcc是个编译驱动器，就像gcc以及LLVM中的clang。
## 1.2 CUDA的内存模型
* Global Memory: 所有threads可以访问
* Shared Memory: 为block内的threads共享
* Local Memory: 属于单个therad
* Constant Memory: 只读内存
* Texture Memory: 只读内存
在RTX 3060上，Global Memory为6GB，Constant Memory为64KB，每个Block的Shared Memory为48KB.

shared memory和registers都在片上，所以速度是最快的。各内存的读取速度从高到底排列为：register file > shared memory > constant memory > texture memory > local memory > global memory.
![CUDA内存模型](pictures/MemorySpacesOnACUDADevice.png)
***
CUDA的内存可以分成`linear memory`和`CUDA arrays`
* linear memory: 使用`cudaMalloc()`分配，使用`cudaFree()`释放，使用`cudaMemcpy()`拷贝。
* CUDA arrays：不透明的内存布局，针对纹理内存的读取进行了优化。