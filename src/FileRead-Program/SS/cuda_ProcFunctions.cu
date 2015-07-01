/**********************************************************************************
Filename	: cuda_ProcFunctions.cu
Authors		: Kevin Wong, Yifan Jian, Marinko Sarunic
Published	: December 6th, 2012

Copyright (C) 2012 Biomedical Optics Research Group - Simon Fraser University
This software contains source code provided by NVIDIA Corporation.

This file is part of a free software. Details of this software has been described 
in the paper titled: 

"GPU Accelerated OCT Processing at Megahertz Axial Scan Rate and High Resolution Video 
Rate Volumetric Rendering"

Please refer to this paper for further information about this software. Redistribution 
and modification of this code is restricted to academic purposes ONLY, provided that 
the following conditions are met:
-	Redistribution of this code must retain the above copyright notice, this list of 
	conditions and the following disclaimer
-	Any use, disclosure, reproduction, or redistribution of this software outside of 
	academic purposes is strictly prohibited

*DISCLAIMER*
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE)ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.
**********************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <cuda.h> //Include the general CUDA Header file
#include <cufft.h> //This is to perform FFT using CUDA
#include <helper_cuda.h> //This is to perform CUDA safecall functions
#include <cuda_runtime.h>
#include "cuda_ProcHeader.cuh" 


typedef float2 Complex;
int numThreadsPerBlock = 256;

bool mallocArrays = false;
bool dcAcquired = true;
bool lambdaCalculated = false;
bool dispPhaseCalculated = false;
enum bufferType { A, B };
bufferType processBufferID = A;


int	frameWidth = 2048;
int frameHeight = 512;
int framesPerBuffer = 1;
int frameCounter = 0;
int bufferSize = frameWidth*frameHeight*framesPerBuffer;
int fftLengthMult = 1;
float minVal;
float maxVal;


float *dev_tempBuffer;
unsigned short *dev_uShortBufferA;
unsigned short *dev_uShortBufferB;


float *dcArray;
Complex *dev_FFTCompBuffer;
cufftHandle fft_plan;
cudaStream_t memcpyStream;
cudaStream_t kernelStream;


/*************************************************************************************************************************/
/*************************************************************************************************************************/
void batchFFT(Complex *d_ComplexArray, cudaStream_t processStream)
{
	cufftExecC2C(fft_plan,
				(cufftComplex *)d_ComplexArray,
				(cufftComplex *)d_ComplexArray,
				CUFFT_FORWARD);
}


/*************************************************************************************************************************
*************************************************************************************************************************/

void initProcCuda()
{
		cudaStreamCreate(&memcpyStream);
		cudaStreamCreate(&kernelStream);
		cudaMalloc((void**)&dev_tempBuffer, bufferSize * sizeof(float));
		cudaMalloc((void**)&dev_uShortBufferA, bufferSize * sizeof(unsigned short));
		cudaMalloc((void**)&dev_uShortBufferB, bufferSize * sizeof(unsigned short));
		cudaMalloc((void**)&dcArray, frameWidth * sizeof(float));
		cudaMemset(dcArray, 0, frameWidth * sizeof(float));
		cudaMalloc((void**)&dev_FFTCompBuffer, bufferSize * fftLengthMult * sizeof(Complex));

		//Be sure to have the fft_width size be dynamic
		cufftPlan1d( &fft_plan, fftLengthMult*frameWidth, CUFFT_C2C, frameHeight *  framesPerBuffer);
		cufftSetStream(fft_plan, kernelStream);
}


void initCudaProcVar(	int frameWid, 
								int frameHei, 
								int framesPerBuff, 
								int fftLenMult)
{
	frameWidth = frameWid;
	frameHeight = frameHei;
	framesPerBuffer = framesPerBuff;
	bufferSize = frameWid*frameHei*framesPerBuff;
	fftLengthMult = fftLenMult;

	numThreadsPerBlock = 256;
	 //65535 is currently the maximum number of threads per kernel
	while (bufferSize/numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, CUDA is unable to handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	printf("Number of Threads Per Block is: %d \n\n", numThreadsPerBlock);
	minVal = 9.5;
	maxVal = 13.0;
}


void cleanUpCUDABuffers()
{
	//Clean up all CUDA Buffers and arryays
	cudaFree(dcArray);
	cudaFree(dev_FFTCompBuffer);
	cudaFree(dev_tempBuffer);

	//Clean up FFT plans created
	cufftDestroy(fft_plan);

	//Clean up the streams created
	cudaStreamDestroy(memcpyStream);
	cudaStreamDestroy(kernelStream);

	mallocArrays = false;
}


void subDC_and_PadComplex(unsigned short *dev_memcpyBuffer, Complex *dev_dstCompBuffer, float *dcArray, cudaStream_t processStream)
{
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX( (bufferSize) / dimBlockX.x);

	if (!dcAcquired) {
		dcAcquireKernel<<<frameWidth/numThreadsPerBlock, numThreadsPerBlock, 0,processStream>>>
				(dev_memcpyBuffer, dcArray, frameWidth, frameHeight);
		dcAcquired = true;
	}

	subDC_PadComplex<<<dimGridX, dimBlockX, 0, processStream>>>
		(dev_memcpyBuffer, dev_dstCompBuffer, dcArray, frameWidth, fftLengthMult*frameWidth);
}


void postFFTDownsize(Complex *d_ComplexArray, float *dev_processBuffer, int frames, int frameIdx, int reduction, cudaStream_t processStream)
{
	int newWidth = frameWidth/reduction;
	int newHeight = frameHeight/reduction;
	float coeff = 1.0f/(maxVal-minVal);
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(newWidth*newHeight*frames/ dimBlockX.x);

	//Downsizing ModLogScale Kernel
	//MLS = Modulus, Log, and Scaling
	//This method of post FFT downsizes the data, and copies into buffer
	//This allows a faster copy rate, full volume viewing range, but lower resolution
	//As opposed to the other method which crops a portion of the whole volume
	downsizeMLS<<<dimGridX, dimBlockX, 0, processStream>>>
		(dev_processBuffer, d_ComplexArray, frameWidth, frameHeight, 
		fftLengthMult*frameWidth, minVal, maxVal, coeff, frameIdx, reduction);
}



//The range var is a portion of the width far, eg width = 1024, a quarter of the width would be the range = 256
void postFFTCrop(Complex *d_ComplexArray, float *dev_processBuffer, int frames, int frameIdx, int offset, int range, cudaStream_t processStream)
{
	float coeff = 1.0f/(maxVal-minVal);
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(range*frameHeight*frames/ dimBlockX.x);

	//MLS = Modulus, Log, and Scaling
	//This method of post FFT crops out a certain portion of the data, and copies into buffer
	//This method preserves resolution, but reduces the viewing range
	//As opposed to the other method which downsizes the whole volume
	cropMLS<<<dimGridX, dimBlockX, 0, processStream>>>
		(dev_processBuffer, d_ComplexArray, frameWidth, frameHeight, 
		fftLengthMult*frameWidth, minVal, maxVal, coeff, frameIdx, offset, range);

}

//This Function calls the kernel which averages the given number of frames into a single frame (B-scan)
void frameAvg(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int numberOfFrames, int frameNum)
{
	if (dev_multiFrameBuff==NULL) {
		dev_multiFrameBuff = dev_tempBuffer;
	}
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height/ dimBlockX.x);
	avgKernel<<<dimGridX, dimBlockX, 0, kernelStream>>>
		(dev_multiFrameBuff, dev_displayBuff, frameNum, numberOfFrames, width*height);
}

//This Kernel will copy one single frame to the display buffer
void copySingleFrame(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int frameNum)
{
	if (dev_multiFrameBuff==NULL) {
		dev_multiFrameBuff = dev_tempBuffer;
	}
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height / dimBlockX.x);
	copySingleFrameFloat<<<dimGridX, dimBlockX, 0, kernelStream>>>
		(dev_multiFrameBuff, dev_displayBuff, frameNum, width*height);
}



void cudaPipeline(	unsigned short *h_buffer, 
								float *dev_frameBuff, 
								int frameIdx,
								int reduction, //This is used only for Downsizing
								int offset, //This is used only for Cropping
								int range) //This is used only for Cropping
{
	//This kernel acts as a GPU synchronization kernel
	//This synchronization prevents any data race conflicts
	//This method of GPU Synchronization has proven to be
	//	the most effective method of synchronization
	//DO NOT remove this kernel!
	syncKernel<<<1,1>>>();
	////

	unsigned short *processBuffer;
	unsigned short *memcpyBuffer;

	if (!mallocArrays) {
		initProcCuda();
		mallocArrays = true;
	}

	if (dev_frameBuff==NULL) {
		dev_frameBuff = dev_tempBuffer;
	}
	
	//Performing dual buffer processing
	//One buffer for memcpy
	//The other buffer for processing
	if (processBufferID==A) {
		processBuffer = dev_uShortBufferA;
		memcpyBuffer = dev_uShortBufferB;
		processBufferID = B;
	} else if (processBufferID==B) {
		processBuffer = dev_uShortBufferB;
		memcpyBuffer = dev_uShortBufferA;
		processBufferID = A;
	}

	//Memcpy data into one buffer
	cudaMemcpyAsync((void *) memcpyBuffer, h_buffer, bufferSize*sizeof(unsigned short), cudaMemcpyHostToDevice, memcpyStream);
	subDC_and_PadComplex(processBuffer, dev_FFTCompBuffer, dcArray, kernelStream);
	batchFFT(dev_FFTCompBuffer, kernelStream);

	//This kernel must be general for 2D OCT, 3D OCT reduce and crop!
	if (reduction==1) {
		postFFTCrop(dev_FFTCompBuffer, dev_frameBuff, framesPerBuffer, frameIdx, offset, range, kernelStream);
	} else {
		postFFTDownsize(dev_FFTCompBuffer, dev_frameBuff, framesPerBuffer, frameIdx, reduction, kernelStream);
	}

	//Another synchronization call explicitly for the streams only
	//This synchronization is a second safety measure over the syncKernel call
	cudaStreamSynchronize(memcpyStream);
}


void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 is the most optimum size for this kernel
	const int blockSize = 128;
	float scaleCoeff = 4.0f/(float)frameWidth;
	int increment = depth;

	while (height*increment>65535) {
		increment >>= 1;
	}

	dim3 dimBlockX(blockSize);
	dim3 dimGridX(height*increment);

	for (int i=0; i<depth; i+=increment) {
		renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
			(dev_volume, dev_fundus, width, scaleCoeff, height*i);
	}
}




/*****************************************************************************************************************************/
/******************************************  Miscellaneous Functions For Adjustments *****************************************/
/*****************************************************************************************************************************/
/*****************************************************************************************************************************/

void acquireDC()
{
	dcAcquired = false;
}
/*****************************************************************************************************************************/
/*****************************************************************************************************************************/
/*****************************************************************************************************************************/
void decreaseMinVal()
{
	minVal -= 0.5f;
	printf("New minVal is %0.1f", minVal);
	printf("\n");
}
/*****************************************************************************************************************************/
void increaseMinVal()
{
	if (minVal==maxVal-1) {
		printf("Error: minVal cannot be equal or greater than maxVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		minVal += 0.5f;
		printf("New minVal is %0.1f", minVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
void increaseMaxVal()
{
	maxVal += 0.5f;
	printf("New maxVal is %0.1f", maxVal);
	printf("\n");
}
/*****************************************************************************************************************************/
void decreaseMaxVal()
{
	if (maxVal==minVal+1) {
		printf("Error: maxVal cannot be equal or less than than minVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		maxVal -= 0.5f;
		printf("New maxVal is %0.1f", maxVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
