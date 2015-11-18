void checkError(cudaError_t err);
void printMatrix(int *mat, int rows, int columns);
IplImage *loadImage(char *path);
void showImageProperties(IplImage *image);
int *getMatrix(IplImage *image, int channel);
int *emptyMatrix(IplImage *image);
IplImage *matrixToIpl(int *b, int *g, int *r, int width, int height);
IplImage *convolution(IplImage *image);


