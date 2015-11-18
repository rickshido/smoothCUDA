/* 
Command syntax:
$convolution <input_image> <output_image>
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cv.h>
#include <highgui.h>
#include "convolution_v3.h"


/*----------------------------- CUDA KERNELS -------------------------------*/
__global__
void conv(int *inImage, int *outImage, int rows, int columns) 
{
	int soma        = 0;
	int maskSize    = 15;
	int padding     = (int) maskSize / 2;
	int initPadding = padding * (-1);
	int endPadding  = padding;
	int divisor     = maskSize * maskSize;
	int i           = threadIdx.y + blockDim.y * blockIdx.y;
	int j           = threadIdx.x + blockDim.x * blockIdx.x;
	int a,b;

	if( (i<(rows-1)) && (j<(columns-1)) && (i>0) && (j>0)) 
	{
		soma = 0;
		for(a=initPadding; a<=endPadding; a++)
		{
			for(b=initPadding; b<=endPadding; b++)
			{
				soma = soma + inImage[(j+a) + (i+b) * columns];
			}
		}
		
		outImage[j + i * columns] = (int) (soma/divisor);
	}
}
/*----------------------- END - CUDA KERNELS -------------------------------*/


int main(int argc, char *argv[]) 
{
	char *inputfile  = argv[1];
	char *outputfile = argv[2];

	IplImage* input  = cvLoadImage(inputfile, CV_LOAD_IMAGE_COLOR);
	IplImage* output;

	clock_t time;

	double time_taken;

	time = clock();

	output = convolution(input);

	time = clock() - time;

	time_taken = ((double)time)/CLOCKS_PER_SEC;
        printf("convolution took %f seconds to execute \n", time_taken);

	cvSaveImage(outputfile, output);

	if(!input)  cvReleaseImage(&input);
	if(!output) cvReleaseImage(&output);

	return 0;
}


IplImage *convolution(IplImage *image) 
{ 
	int *h_inMatrix, *b_outMatrix, *g_outMatrix, *r_outMatrix;
	int *d_inMatrix, *d_outMatrix;

	int buffer_size;
	int m, n;

	cudaError_t err;

	m = image->height/2;
	n = image->width;

	buffer_size = sizeof(int) * image->width * image->height/2;

	dim3 DimGrid((n-1)/16+1,(m-1)/16+1,1);
	dim3 DimBlock(16,16,1);

	int i;


		//BLUE
		h_inMatrix = getMatrix(image, 0);
		b_outMatrix = emptyMatrix(image);

		cudaMalloc((void**) &d_inMatrix, buffer_size);
		cudaMalloc((void**) &d_outMatrix, buffer_size);

		cudaMemcpy(d_inMatrix,  h_inMatrix,  buffer_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_outMatrix, b_outMatrix, buffer_size, cudaMemcpyHostToDevice);

		conv<<<DimGrid, DimBlock>>>(d_inMatrix, d_outMatrix, image->height, image->width);	
	
		cudaMemcpy(b_outMatrix, d_outMatrix, buffer_size, cudaMemcpyDeviceToHost);

		cudaFree(d_inMatrix);
		cudaFree(d_outMatrix);
		free(h_inMatrix);

		//Green
		h_inMatrix = getMatrix(image, 1);
       	 	g_outMatrix = emptyMatrix(image);

        	cudaMalloc((void**) &d_inMatrix, buffer_size);
        	cudaMalloc((void**) &d_outMatrix, buffer_size);

        	cudaMemcpy(d_inMatrix,  h_inMatrix,  buffer_size, cudaMemcpyHostToDevice);
        	cudaMemcpy(d_outMatrix, g_outMatrix, buffer_size, cudaMemcpyHostToDevice);

        	conv<<<DimGrid, DimBlock>>>(d_inMatrix, d_outMatrix, image->height, image->width);

        	cudaMemcpy(g_outMatrix, d_outMatrix, buffer_size, cudaMemcpyDeviceToHost);

        	cudaFree(d_inMatrix);
        	cudaFree(d_outMatrix);
        	free(h_inMatrix);

		//RED
		h_inMatrix = getMatrix(image, 2);
        	r_outMatrix = emptyMatrix(image);

        	cudaMalloc((void**) &d_inMatrix, buffer_size);
        	cudaMalloc((void**) &d_outMatrix, buffer_size);

        	cudaMemcpy(d_inMatrix,  h_inMatrix,  buffer_size, cudaMemcpyHostToDevice);
        	cudaMemcpy(d_outMatrix, r_outMatrix, buffer_size, cudaMemcpyHostToDevice);

        	conv<<<DimGrid, DimBlock>>>(d_inMatrix, d_outMatrix, image->height, image->width);

        	cudaMemcpy(r_outMatrix, d_outMatrix, buffer_size, cudaMemcpyDeviceToHost);

        	cudaFree(d_inMatrix);
        	cudaFree(d_outMatrix);
        	free(h_inMatrix);


	return matrixToIpl(b_outMatrix, g_outMatrix, r_outMatrix, image->width, image->height);
}


void checkError(cudaError_t err) 
{ 
	if(err != cudaSuccess) 
	{
		printf("CUDA error:\n");
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}


void printMatrix(int *mat, int rows, int columns) 
{
	int i,j;

	for(i=0; i<rows; i++) 
	{
		for(j=0; j<columns; j++) 
		{
			printf("%.2f ", mat[j + i*columns]);
		}
		printf("\n");
	}
}


IplImage *loadImage(char *path) 
{
	IplImage *image = cvLoadImage(path, -1);

	if(!image) 
	{
		printf("\nError on load image: %s", path);
		exit(EXIT_FAILURE);
	}

	return image;
}


void showImageProperties(IplImage *image) 
{
	if(image) 
	{
		printf("Width: %d\n", image->width);
		printf("Height: %d\n", image->height);
		printf("Channels: %d\n", image->nChannels);
	} else {
		printf("Image is NULL\n");
	}
}


int *getMatrix(IplImage *image, int channel) 
{
	int i, j;

	int *matrix = (int*) malloc(sizeof(int) * image->width * image->height);

	for( i = 0; i < image->height; i++ ) 
		for( j = 0; j < image->width; j++ ) 
			matrix[i * image->width + j] = cvGet2D(image, i, j).val[channel];

	return matrix;
}


int *emptyMatrix(IplImage *image)
{
	return (int*) malloc(sizeof(int) * image->width * image->height);
}


IplImage *matrixToIpl(int *b, int *g, int *r, int width, int height) 
{
	int i, j;
	CvScalar pixel;

	IplImage *image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	for( i = 0; i < height; i++ )
	{
		for( j = 0; j < width; j++ )
		{
			cvSet2D(image, i, j, cvScalar(b[i * width + j], g[i * width + j], r[i * width + j], 0));
		}
	}

	return image;
}
