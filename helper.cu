#include "helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 1024
#define NAME_BUFFER_SIZE 256
#define MAX_FILTER_SIZE 20
#define INITAL_BLOCK_SIZE 4

float* flatenFilter(float** map, const int s)
{
	float* arr = (float*)malloc(sizeof(float) * s * s);
	if (!arr)
		exit(STATUS_MEMEORY_ALLOCATION_ERROR);
	for (int i = 0, k = 0; i < s; i++)
		for (int j = 0; j < s; j++)
			arr[k++] = map[i][j];
	return arr;
}

int showMenu(char** names, int count)
{
	puts("CUDA conv");
	puts("<Lea Mladenic, Dino Miletic, Ivan Kulis>");
	puts("Odaberite filter:");
	int pick = FIRST_RUN;
	
	while (pick < 0 || pick >= count) {
		if (pick != FIRST_RUN)
			puts("Bad input!");
		for (int i = 0; i < count; i++) {
			printf("[%d] %s\n", i, names[i]);
		}
		printf("Input: ");
		scanf("  %d", &pick);
	}
	return pick;
}

int countChar(char* str, int size, char del) {
	int count = 0;
	for (int i = 0; i < size; i++) {
		if (str[i] == '\0' || str[i] == '\n')
			return count;
		if (str[i] == del)
			count++;
	}
}

void readFilters(char* file, float**** out_filters, int** out_sizes, char*** out_names, int* count) {
	int resizeCounter = 1;
	char** names = (char**)malloc(sizeof(char*) * INITAL_BLOCK_SIZE);
	float*** filters = (float***)malloc(sizeof(float**) * INITAL_BLOCK_SIZE);
	int* sizes = (int*)malloc(sizeof(int) * INITAL_BLOCK_SIZE);
	FILE* fp = fopen(file,"r");
	if (!fp) {
		exit(STATUS_OPEN_FILE_ERROR);
	}
	char buffer[BUFFER_SIZE];
	char nameBuffer[BUFFER_SIZE];
	int counter = -1;
	int row = 0;
	int size = 0;
	float coef = 1.0;
	while (!feof(fp)) {
		if (counter >= INITAL_BLOCK_SIZE * resizeCounter) {
			char** tmp;
			float*** ftmp;
			tmp = (char**)realloc(names, sizeof(char*) * INITAL_BLOCK_SIZE * ++resizeCounter);
			ftmp = (float***)realloc(filters, sizeof(float**) * INITAL_BLOCK_SIZE * resizeCounter);
			if (!tmp || !ftmp) {
				free(names);
				free(filters);
				exit(STATUS_MEMEORY_ALLOCATION_ERROR);
			}
			names = tmp;
			filters = ftmp;
		}
		if (fgets(buffer, BUFFER_SIZE, fp) == 0)
			exit(STATUS_READ_ERROR);
		if (buffer[0] == '#') // comment, skip
			continue;
		if (buffer[0] == '*') { // definition
			row = 0;
			counter++;
			sscanf(buffer, "*%s %d %f", &nameBuffer, &size, &coef);
			sizes[counter] = size;
			names[counter] = (char*)malloc(sizeof(char) * NAME_BUFFER_SIZE + 1);
			strcpy(names[counter], nameBuffer);

			filters[counter] = (float**)malloc(sizeof(float**) * size);
			for (int i = 0; i < size; i++)
				filters[counter][i] = (float*)malloc(sizeof(float*) * size);
			continue;
		}
		filters[counter][row] = (float*) malloc(sizeof(float) * size);
		int off = 0;
		char* ptr = buffer;
		for (int i = 0; i < size; i++) {
			sscanf(ptr, "%f%n,", &filters[counter][row][i], &off);
			filters[counter][row][i] *= coef;
			ptr += off+1;
		}
		row++;
	}
	*(out_filters) = filters;
	*(out_sizes) = sizes;
	*(out_names) = names;
	*count = counter;
}