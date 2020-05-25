#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LOADBMP_IMPLEMENTATION

#include <stdio.h>
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

__global__ void convolution(byte_t* pixelMap, char* filter, double k, byte_t* resultMap, int width, int height) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= width && j >= height)
		return;
	// printf("%d %d %d\n", i, j, z);
	// C[i][j] = A[i][j] + B[i][j];
	char r[FILTER_SIZE][FILTER_SIZE]; // result block
	for (int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++) {
			r[x + 1][y + 1] = (i + x > width || i + x < 0 || y + j > height || y + j < 0)
				? 0
				: *(&pixelMap[(i + x) * height + (j + y)] + z) * filter[FILTER_SIZE * (x + 1) + (j + 1)];
		}
	double sum = 0;
	for (int x = 0; x < 3; x++)
		for (int y = 0; y < 3; y++)
			sum += r[x][y];
	sum *= k;
	// printf("%c %d", (byte_t)sum, (int)sum);
	*(&resultMap[i * height + j] + z) = *(&pixelMap[i * height + j] + z);
}

double coef[2] = { 1 / 9, 1.0};
char filters[2][3][3] = {
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
	char* flatFilter = (char*)flattenArray((void**)filters[BoxBlur], 3, 3, sizeof(char));

	byte_t* d_pixelMap, *d_resultMap, *h_resultMap;
	char* d_filter;
	gpuErrchk(cudaMalloc((void**)&d_filter, sizeof(char) * 3 * 3));
	gpuErrchk(cudaMalloc((void**)&d_pixelMap, sizeof(byte_t) * size));
	gpuErrchk(cudaMalloc((void**)&d_resultMap, sizeof(byte_t) * size));
	//---cpy

	gpuErrchk(cudaMemcpy(d_pixelMap, pixels, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_filter, flatFilter, sizeof(char) * 3 * 3, cudaMemcpyHostToDevice));
	//DO STUFF
	/*
	Declare and allocate host and device memory. <
	Initialize host data. <
	Transfer data from the host to the device. <
	Execute one or more kernels. <
	Transfer results from the device to the host. <
	*/
	dim3 threadsPerBlock(16, 16, 3);
	dim3 numberOfBlocks(width / 8, height / 8);
	convolution <<<numberOfBlocks, threadsPerBlock >>> (d_pixelMap, d_filter, coef[BoxBlur], d_resultMap, width, height);

	gpuErrchk(cudaPeekAtLastError());
	h_resultMap = (byte_t*)malloc(sizeof(byte_t) * size);

	gpuErrchk(cudaMemcpy(h_resultMap, d_resultMap, size, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	loadbmp_encode_file("lena2.bmp", h_resultMap, width, height, LOADBMP_RGB);
	
	free(pixels);
	free(h_resultMap);
	free(flatFilter);
	cudaFree(d_filter);
	cudaFree(d_resultMap);
	cudaFree(d_pixelMap);
	return 0;
}