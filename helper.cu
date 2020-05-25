#include "helper.h"
#include <stdlib.h>
#include <string.h>

char** d2(char* arr, const int w, const int h) {
	char** map = (char**)malloc(sizeof(char*) * w);
	int i, j;
	for (i = 0; i < w; i++) {
		map[i] = (char*)malloc(sizeof(char) * h + 1);

		for (j = 0; j < h; j++)
			map[i][j] = arr[i * h + j];
		map[i][j] = '\0';
	}
	return map;
}

char* d1(char** map, const int w, const int h) {
	const int size = w * h;
	char* arr = (char*)malloc(sizeof(char) * size + 1);
	if (!arr) {
		exit(-1);
	}
	int i, j;
	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++)
			arr[i * h + j] = map[i][j];
	}
	arr[size] = '\0';

	return arr;
}


unsigned char** pixelArrayToPixelMap(unsigned char* arr, const int w, const int h) {
	unsigned char** map = (unsigned char**)malloc(sizeof(char*) * w);
	if (!map) {
		exit(-1);
	}
	for (int i = 0; i < w; i++) {
		map[i] = (unsigned char*)malloc(sizeof(unsigned char) * h);

		for (int j = 0; j < h; j++)
			map[i][j] = arr[i * h + j];
	}
	return map;
}

unsigned char* mapToPixelArray(unsigned char** map, const int w, const int h)
{
	const int size = w * h;
	unsigned char* arr = (unsigned char*)malloc(sizeof(unsigned char) * size);
	if (!arr) {
		exit(-1);
	}

	for (int i = 0; i < w; i++) {
		for(int j = 0; j < h; j++)
			arr[i * h + j] = map[i][j];
	}

	return arr;
}

void* flattenArray(void** map, const int w, const int h, const size_t elementSize)
{
	const int size = w * h;
	char* arr = (char*) malloc(elementSize * size);
	if (!arr) {
		exit(-1);
	}

	for (int i = 0; i < w; i++) {
		memcpy(&arr[i], &map[i], h * elementSize);
	}
	return (void*)arr;
}
