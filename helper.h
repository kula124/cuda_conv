// typedef unsigned char byte_t;
typedef unsigned int imgsize_t;

// unsigned char** pixelArrayToPixelMap(unsigned char* arr, const int w, const int h);
unsigned char* mapToPixelArray(unsigned char** arr, const int w, const int h);
void* flattenArray(void** arr, const int w, const int h, const size_t elementSize);
char** d2(char* arr, const int w, const int h);
char* d1(char** map, const int w, const int h);