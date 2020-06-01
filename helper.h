#ifndef HELPER_H
#define HELPER_H
#define STATUS_MEMEORY_ALLOCATION_ERROR -1
#define STATUS_OPEN_FILE_ERROR -2
#define STATUS_OK 0
#define STATUS_READ_ERROR -3
void readFilters(char* file, float**** out_filters, int** out_sizes, char*** out_names, int* count);
void handleError(int status);
float* flatenFilter(float**, const int s);
#endif // !HELPER_H