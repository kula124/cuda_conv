#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageHandler.h"
#include "ImageModel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "helper.h"

#define BLOCK_SIZE 32 // ideal blocksize for preformance: 32 x 32 = 1024 => block core count
enum Filter
{
	BoxBlur = 0,
	Sharpen = 1
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}
typedef unsigned char byte_t;
__global__ void convolution(float* pixelMap, float* filter, float* resultMap, int width, int height, int components, const int FILTER_SIZE) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int filterRadius = FILTER_SIZE / 2;
	if (i >= width || j >= height)
		return;
	for (int z = 0; z < components; z++) {// itterate thru colors
		float sum = 0.0;
		for (int x = -filterRadius; x <= filterRadius; x++) // itterate thru filter rows
			for (int y = -filterRadius; y <= filterRadius; y++) // itterate thru filter cols
				sum += (i + x >= width || i + x < 0 || y + j >= height || y + j < 0)
					? 0 // edge ignore solution
					: filter[(x + 1) * FILTER_SIZE + (y + 1)] // filter x pixel[color]
						* pixelMap[((i + x) * width + (j + y)) * components + z];
		resultMap[(i * width + j) * components + z] = sum;
	}
}

int main(char** argv, int argc) {
	float* d_pixelMap, * d_resultMap, * h_resultMap, *** filters;
	char** filter_names;
	int* filter_sizes, filter_count;
	int size;
	readFilters("filters.txt",&filters,&filter_sizes,&filter_names, &filter_count);
	int pick = showMenu(filter_names, filter_count);
	const int FILTER_SIZE = filter_sizes[pick];

	auto inputImage = importPPM("lena.ppm");
	auto outputImage = Image_new(inputImage->width, inputImage->height, inputImage->channels);
	size = inputImage->width * inputImage->height * inputImage->channels;
	float* flatFilter = flatenFilter(filters[pick], FILTER_SIZE);
	float* d_filter;
	/*
	Declare and allocate host and device memory. <
	Initialize host data. <
	Transfer data from the host to the device. <
	Execute one or more kernels. <
	Transfer results from the device to the host. <
	*/
	// malloc
	gpuErrchk(cudaMalloc((void**)&d_filter, sizeof(float) * FILTER_SIZE * FILTER_SIZE));
	gpuErrchk(cudaMalloc((void**)&d_pixelMap, sizeof(float) * size));
	gpuErrchk(cudaMalloc((void**)&d_resultMap, sizeof(float) * size));
	//---cpy
	gpuErrchk(cudaMemcpy(d_pixelMap, inputImage->data, size * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_filter, flatFilter, sizeof(float) * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice));
	//DO STUFF

	dim3 numberOfBlocks(ceil(inputImage->width) / BLOCK_SIZE, ceil(inputImage->height / BLOCK_SIZE));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	// cudaDeviceSynchronize();
	convolution << <numberOfBlocks, threadsPerBlock >> > (d_pixelMap, d_filter, d_resultMap, inputImage->width, inputImage->height, 3, FILTER_SIZE);

	gpuErrchk(cudaPeekAtLastError());
	h_resultMap = (float*)malloc(sizeof(float) * inputImage->width * inputImage->height * inputImage->channels);

	gpuErrchk(cudaMemcpy(h_resultMap, d_resultMap, size * sizeof(float), cudaMemcpyDeviceToHost));

	outputImage->data = h_resultMap;
	exportPPM("output.ppm", outputImage);

	//clean up
	free(h_resultMap);
	free(flatFilter);
	cudaFree(d_filter);
	cudaFree(d_resultMap);
	cudaFree(d_pixelMap);
	return 0;
}