#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LOADBMP_IMPLEMENTATION
#include "ImageHandler.h"
#include "ImageModel.h"
#include <stdio.h>
#include <math.h>
#include "bmpLoader.h"
#include "helper.h"

#define FILTER_SIZE 3
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
		if (abort) exit(code);
	}
}

__global__ void convolution(float* pixelMap, int* filter, double coef, float* resultMap, int width, int height, int components) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float f[] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
	// double f[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
	// double f[] = {-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,};
	if (i >= width || j >= height)
		return;
	for (int z = 0; z < components; z++) {// itterate thru channels
		float sum = 0.0;
		for (int x = -(FILTER_SIZE / 2); x <= (FILTER_SIZE / 2); x++) // itterate thru filter rows
			for (int y = -(FILTER_SIZE / 2); y <= (FILTER_SIZE / 2); y++) { // itterate thru filter cols
				float pixel = pixelMap[((i + x) * width + (j + y)) * components + z];
				float ff = f[(x + 1) * FILTER_SIZE + (y + 1)];
				sum += (i + x >= width || i + x < 0 || y + j >= height || y + j < 0)
					? 0
					: ff * pixel;
			}
		resultMap[(i * width + j) * components + z] = sum;
	}
}

// double coef[2] = { 1, 1.0 };
int filters[2][3][3] = {
	{
		{1,1,1},
		{1,1,1},
		{1,1,1}
	},
	{
		{0,-1,0},
		{-1,5,-1},
		{0,-1,0}
	}
};

int main(char** argv, int argc) {
	float* d_pixelMap, * d_resultMap, * h_resultMap;
	int size;
	auto inputImage = importPPM("lena.ppm");
	exportPPM("test.ppm", inputImage);
	auto outputImage = Image_new(inputImage->width, inputImage->height, inputImage->channels);
	size = inputImage->width * inputImage->height * inputImage->channels * sizeof(float);
	//int* flatFilter = (int*)flattenArray((void**)filters[BoxBlur], 3, 3, sizeof(int));
	int flatFilter[] = { 0,-1,0,-1,5,-1,0,-1,0 };
	// float* d_pixelMap, *d_resultMap, *h_resultMap;
	int* d_filter;
	//gpuErrchk(cudaMalloc((void**)&d_filter, sizeof(int) * 3 * 3));
	gpuErrchk(cudaMalloc((void**)&d_pixelMap, sizeof(float) * size));
	gpuErrchk(cudaMalloc((void**)&d_resultMap, sizeof(float) * size));
	//---cpy

	gpuErrchk(cudaMemcpy(d_pixelMap, inputImage->data, size, cudaMemcpyHostToDevice));
	// gpuErrchk(cudaMemcpy(d_filter, flatFilter, sizeof(int) * 9, cudaMemcpyHostToDevice));
	//DO STUFF
	/*
	Declare and allocate host and device memory. <
	Initialize host data. <
	Transfer data from the host to the device. <
	Execute one or more kernels. <
	Transfer results from the device to the host. <
	*/
	printf("%d %d\n", inputImage->width, inputImage->height);
	dim3 numberOfBlocks(512 / 32, 512 / 32);
	dim3 threadsPerBlock(32, 32);
	cudaDeviceSynchronize();
	convolution << <numberOfBlocks, threadsPerBlock >> > (d_pixelMap, NULL, 0.0, d_resultMap, inputImage->width, inputImage->height, 3);

	gpuErrchk(cudaPeekAtLastError());
	h_resultMap = (float*)malloc(sizeof(float) * 512 * 512 * 3);

	gpuErrchk(cudaMemcpy(h_resultMap, d_resultMap, size, cudaMemcpyDeviceToHost));

	// cudaDeviceSynchronize();
	// loadbmp_encode_file("lena2.bmp", h_resultMap, width, height, LOADBMP_RGB);
	outputImage->data = h_resultMap;
	exportPPM("output.ppm", outputImage);

	free(h_resultMap);
	// free(flatFilter);
	// fclose(fp);
	// cudaFree(d_filter);
	cudaFree(d_resultMap);
	cudaFree(d_pixelMap);
	return 0;
}