#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*
This was a learning project in which I was using thread blocks 
and a very large number of threads (40 Million) to compute atomically.
*/



using namespace std;


__global__ void divide(int *buff)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i = index % 10;
	
	atomicAdd(&buff[i], i);

}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	const unsigned int ThreadNum = 40000000;
	const unsigned int BlockWidth = 1000;
	
	const int ArraySize = 10;
	const int ArrayBytes = ArraySize * sizeof(int);

	int hostArray[ArraySize];

	int * deviceArray;
	cudaMalloc((void **)&deviceArray, ArrayBytes);
	cudaMemset((void *)deviceArray, 0, ArrayBytes);
	cudaEventRecord(start);
	//ThreadNum / BlockWidth = 40,000 Thread Blocks.  BlockWidth = 1000, which means 1000 Threads for each Thread Block, 40,000 x 1,000 = 40,000,000;
	divide << <ThreadNum / BlockWidth, BlockWidth >> > (deviceArray);
	cudaEventRecord(stop);
	cudaMemcpy(hostArray, deviceArray, ArrayBytes, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 10; i++) {
		cout << hostArray[i] << endl;
	}

	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	cout <<"Time: " << ms << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(deviceArray);
	return 0;

}
