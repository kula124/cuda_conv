#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LOADBMP_IMPLEMENTATION

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

// FILE* fp = fopen("./wtf.txt", "w+");

typedef unsigned char byte_t;
__global__ void convolution(byte_t* pixelMap, int* filter, double coef, byte_t* resultMap, int width, int height, int components) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	double f[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
	// double f[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
	// double f[] = {-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,};
	if (i >= width || j >= height)
		return;
	double sum = 0.0;
	static int c = 10;
	for (int z = 0; z < components; z++) {// itterate thru channels
		for (int x = -(FILTER_SIZE / 2); x <= (FILTER_SIZE / 2); x++) // itterate thru filter rows
			for (int y = -(FILTER_SIZE / 2); y <= (FILTER_SIZE / 2); y++) { // itterate thru filter cols
				byte_t pixel = pixelMap[((i + x) * width + (j + y)) * components + z];
				//byte_t pixel = pixelMap[z * (height * width) + (i + x) * height + (j + y)];
				double ff = f[(x + 1) * FILTER_SIZE + (y + 1)];

				if (i == 19 && j == 32)
					printf("PM:%u * f:%lf(%d|%d) = %lf\n",pixel, ff,x,y, ff * pixel);
				sum += (i + x >= width || i + x < 0 || y + j >= height || y + j < 0)
					? 0
					: ff * (int)pixel;
			}
		if (sum <= 0)
			sum += 125;
		if (i == 19 && j == 32)
			printf("sum: %lf| new pixel component: [%u] | oldPixel: [%u]\n", sum, (byte_t)((int)(sum)), pixelMap[(i * width + j) * components + z]);
		sum *= coef;
		// resultMap[z * (height * width) + i * height + j] = (byte_t)((int)ceil(sum));
		resultMap[(i * width + j) * components + z] = (byte_t)((int)ceil(sum));
		sum = 0.0;
	}
	// sum *= coef;
	// *(&resultMap[i * height + j] + z) = (byte_t)((int)ceil(sum));//*(&pixelMap[i * height + j] + z);
}

double coef[2] = { 1, 1.0 };
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
	byte_t* pixels = NULL;
	imgsize_t width = 0, height = 0, size = 0;
	if (loadbmp_decode_file("lena.bmp", &pixels, &width, &height, LOADBMP_RGB))
	{
		puts("Error!");
		return 1;
	}
	size = width * height * 3;
	//int* flatFilter = (int*)flattenArray((void**)filters[BoxBlur], 3, 3, sizeof(int));
	int flatFilter[] = { 0,-1,0,-1,5,-1,0,-1,0 };
	byte_t* d_pixelMap, *d_resultMap, *h_resultMap;
	int* d_filter;
	gpuErrchk(cudaMalloc((void**)&d_filter, sizeof(int) * 3 * 3));
	gpuErrchk(cudaMalloc((void**)&d_pixelMap, sizeof(byte_t) * size));
	gpuErrchk(cudaMalloc((void**)&d_resultMap, sizeof(byte_t) * size));
	//---cpy

	gpuErrchk(cudaMemcpy(d_pixelMap, pixels, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_filter, flatFilter, sizeof(int) * 3 * 3, cudaMemcpyHostToDevice));
	//DO STUFF
	/*
	Declare and allocate host and device memory. <
	Initialize host data. <
	Transfer data from the host to the device. <
	Execute one or more kernels. <
	Transfer results from the device to the host. <
	*/
	dim3 numberOfBlocks(width / 32, height / 32);
	dim3 threadsPerBlock(32, 32);
	convolution <<<numberOfBlocks, threadsPerBlock >>> (d_pixelMap, d_filter, coef[BoxBlur], d_resultMap, width, height, LOADBMP_RGB);

	gpuErrchk(cudaPeekAtLastError());
	h_resultMap = (byte_t*)malloc(sizeof(byte_t) * size);

	gpuErrchk(cudaMemcpy(h_resultMap, d_resultMap, size, cudaMemcpyDeviceToHost));

	// cudaDeviceSynchronize();
	loadbmp_encode_file("lena2.bmp", h_resultMap, width, height, LOADBMP_RGB);
	
	free(pixels);
	free(h_resultMap);
	// free(flatFilter);
	// fclose(fp);
	cudaFree(d_filter);
	cudaFree(d_resultMap);
	cudaFree(d_pixelMap);
	return 0;
}