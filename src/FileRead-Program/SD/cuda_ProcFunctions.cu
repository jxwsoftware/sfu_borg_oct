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
#include <cutil_inline.h> //This is to perform CUDA safecall functions
#include <cuda_runtime.h>
#include "cuda_ProcKernels.cu"

const float MINVAL = 7.40f;
const float MAXVAL = 9.80f;
const float MINVAL_WITH_PREFILTER = 18.2f;
const float MAXVAL_WITH_PREFILTER = 21.0f;

typedef float2 Complex;
int numThreadsPerBlock = 256;

bool mallocArrays = false;
bool dcAcquired = true;
bool lambdaCalculated = false;
bool dispPhaseCalculated = false;
enum bufferType { A, B };
bufferType processBufferID = A;


float *d_floatFrameBuffer;

int	frameWidth = 2048;
int frameHeight = 512;
int framesPerBuffer = 1;
int frameCounter = 0;
int bufferSize = frameWidth*frameHeight*framesPerBuffer;
int fftLengthMult = 1;
float minVal = MINVAL_WITH_PREFILTER;
float maxVal = MAXVAL_WITH_PREFILTER;
float scalingFactor = 1.0;
float dispMagn;
float dispVal;
float dispValThird;
float lambdaMin;
float lambdaMax;

// 0 for linear, 1 for cubic spline, 2 for prefiltered spline+
int samplingMethod = 0; 


//Memory Pointers required for CUDA Processing
unsigned short *dev_ushortBuffer;
float *dev_floatBuffer;
float *dev_tempBuffer;
float *lookupLambda;
float *linearIdx;
float *dcArray;
Complex *dispPhaseCartesian;
Complex *dev_FFTCompBuffer;
Complex *dev_CompBuffer;

//Create the FFT and Hilbert Transform Plans
cufftHandle fft_plan;
cufftHandle hilbert_plan;

cudaStream_t memcpyStream;
cudaStream_t kernelStream;
cudaEvent_t start_event, stop_event;

/*************************************************************************************************************************/
extern "C" void callCubicPrefilter(float *dev_Coeffs, int pitch, int width, int height, int threadPerBlock, cudaStream_t processStream);
extern "C" void initUshortTexture(unsigned short *host_array, int width, int height, int numFrames, cudaStream_t thisStream);
extern "C" void initFloatTexture(float *dev_array, int width, int height, int numFrames, cudaStream_t thisStream);
extern "C" void printComplexArray(Complex *devArray);
extern "C" void printFloatArray(float *devArray);
extern "C" void printUShortArray(unsigned short *devArray);
extern "C" void cleanUpCudaArray();

/*************************************************************************************************************************/


//Interpolation Kernel is NO LONGER STREAMED, to avoid conflict with texture setup
void interp_and_DCSub(Complex *dev_complexBuffer)
{

	if (!lambdaCalculated) {
		int scale = int(frameWidth-1);
		float diff = lambdaMax - lambdaMin;
		float *h_lookupLambda = (float *)malloc(frameWidth * sizeof(float));
		float *h_linearIdx = (float *)malloc(frameWidth * sizeof(float));
		
		for (int i=0; i<frameWidth; i++) {
			//This is the equation to determine the lookup coordinates for the lambda-pixel domain
			h_linearIdx[i] = float(i-frameWidth/2 + 1);
			h_lookupLambda[i] = ((lambdaMin*lambdaMax*scale)/(lambdaMin*i + lambdaMax*(scale-i)) - lambdaMin) * scale/diff;
		}

		cutilSafeCall( cudaMemcpy((void *) linearIdx, h_linearIdx, frameWidth*sizeof(float), cudaMemcpyHostToDevice));
		cutilSafeCall( cudaMemcpy((void *) lookupLambda, h_lookupLambda, frameWidth*sizeof(float), cudaMemcpyHostToDevice));

		free(h_lookupLambda);
		free(h_linearIdx);

		lambdaCalculated = true;
	}

	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(( bufferSize )/ dimBlockX.x);
	interp_DCSub<<<dimGridX, dimBlockX>>>
		(samplingMethod, frameWidth, fftLengthMult*frameWidth, frameHeight, lookupLambda, dcArray, dcAcquired, dev_complexBuffer);

	if (!dcAcquired) {
		dcAcquireKernel<<<frameWidth/numThreadsPerBlock, numThreadsPerBlock>>>
			(dev_complexBuffer, dcArray, frameWidth, frameHeight);

		dcSubKernel<<<dimGridX, dimBlockX>>>(dev_complexBuffer, dcArray, frameWidth);
		dcAcquired = true;
	}
}


//Subtract DC Must use Complex for Source Float, Because our Host to Array implementation requires this to be so
void subtractDC(Complex *dev_complexBuffer, float *dcArray, cudaStream_t processStream)
{
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(( bufferSize )/ dimBlockX.x);
	if (!dcAcquired) {
		dcAcquireKernel<<<frameWidth/numThreadsPerBlock, numThreadsPerBlock, 0,processStream>>>
			(dev_complexBuffer, dcArray, frameWidth, frameHeight);
		dcAcquired = true;
	}
	dcSubKernel<<<dimGridX, dimBlockX, 0, processStream>>>(dev_complexBuffer, dcArray, frameWidth);
}


//Dispersion Compensation Includes: Hilbert Transform and Phase Multiplication
void dispersionCompensation(Complex *d_ComplexArray, Complex *d_fftCompBuffer, cudaStream_t processStream)
{
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(( bufferSize )/ dimBlockX.x);

//First Perform the Hilbert Transform
	cufftSafeCall(
        cufftExecC2C(hilbert_plan,
                     (cufftComplex *)d_ComplexArray,
                     (cufftComplex *)d_ComplexArray,
                     CUFFT_FORWARD)
	);

	hilbertCoeff<<<dimGridX, dimBlockX, 0, processStream>>>
		(d_ComplexArray, frameWidth);

	cufftSafeCall(
        cufftExecC2C(hilbert_plan,
                     (cufftComplex *)d_ComplexArray,
                     (cufftComplex *)d_ComplexArray,
                     CUFFT_INVERSE)
	);
//END OF Hilbert Transform
	
	if (!dispPhaseCalculated) {
		getDispersion<<<frameWidth/numThreadsPerBlock, numThreadsPerBlock, 0, processStream>>>
			(dispVal, dispValThird, dispMagn, linearIdx, dispPhaseCartesian, frameWidth);
		dispPhaseCalculated = true;
	}

	//This Kernel Will multiply values by Dispersion Phase AND will copy to Padded Array
	dispComp_and_PadComplex<<<dimGridX, dimBlockX, 0, processStream>>>
		(d_ComplexArray, dispPhaseCartesian, d_fftCompBuffer, frameWidth, fftLengthMult*frameWidth);
}
// End of Dispersion Compensation


void batchFFT(Complex *d_ComplexArray, cudaStream_t processStream)
{
	cufftSafeCall(
		cufftExecC2C(fft_plan,
					(cufftComplex *)d_ComplexArray,
					(cufftComplex *)d_ComplexArray,
					CUFFT_FORWARD)
	);
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
	//Thsi allows a faster copy rate, but lower resolution
	//As opposed to the other method which crops the whole volume
	downsizeMLS<<<dimGridX, dimBlockX, 0, processStream>>>
		(dev_processBuffer, d_ComplexArray, frameWidth, frameHeight, 
		fftLengthMult*frameWidth, minVal, maxVal, coeff, frameIdx, reduction);
}



//The range var is a portion of the width far, eg width = 1024, a quarter of the width would be the range = 256
void postFFTCrop(Complex *d_ComplexArray, float *dev_processBuffer, int frames, int frameIdx, int offset, int range, cudaStream_t processStream)
{
	//printf("%d ", range);
	float coeff = 1.0f/(maxVal-minVal);

	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(range*frameHeight*frames/ dimBlockX.x);

	//MLS = Modulus, Log, and Scaling
	//This method of post FFT crops out a certain portion of the data, and copies into buffer
	//This method conserves resolution, but cuts down on the viewing range
	//As opposed to the other method which downsizes the whole volume
	cropMLS<<<dimGridX, dimBlockX, 0, processStream>>>
		(dev_processBuffer, d_ComplexArray, frameWidth, frameHeight, 
		fftLengthMult*frameWidth, minVal, maxVal, coeff, frameIdx, offset, range);

}


/*************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
*************************************************************************************************************************/

void initProcCuda()
{
		cutilSafeCall( cudaMalloc((void**)&dev_tempBuffer, bufferSize * sizeof(float)));

		cudaStreamCreate(&memcpyStream);
		cudaStreamCreate(&kernelStream);

		cutilSafeCall( cudaMalloc((void**)&dev_ushortBuffer, bufferSize * sizeof(unsigned short)));
		cudaMemset( dev_ushortBuffer, 0, bufferSize * sizeof(unsigned short));
		cutilSafeCall( cudaMalloc((void**)&dev_floatBuffer, bufferSize * sizeof(float)));
		cudaMemset( dev_floatBuffer, 0, bufferSize * sizeof(float));

		cutilSafeCall( cudaMalloc((void**)&dcArray, frameWidth * sizeof(float)));
		cutilSafeCall( cudaMalloc((void**)&lookupLambda, frameWidth * sizeof(float)));
		cutilSafeCall( cudaMalloc((void**)&linearIdx, frameWidth * sizeof(float)));
		cutilSafeCall( cudaMalloc((void**)&dispPhaseCartesian, frameWidth * sizeof(Complex)));
		cudaMemset(dcArray, 0, frameWidth * sizeof(float));
		cudaMemset(lookupLambda, 0, frameWidth * sizeof(float));
		cudaMemset(linearIdx, 0, frameWidth * sizeof(float));
		cudaMemset(dispPhaseCartesian, 0, frameWidth * sizeof(float));

		cutilSafeCall( cudaMalloc((void**)&dev_CompBuffer, bufferSize * sizeof(Complex)));
		cutilSafeCall( cudaMalloc((void**)&dev_FFTCompBuffer, bufferSize * fftLengthMult * sizeof(Complex)));

		//Be sure to have the fft_width size be dynamic
		cufftSafeCall( cufftPlan1d( &fft_plan, fftLengthMult*frameWidth, CUFFT_C2C, frameHeight *  framesPerBuffer));
		cufftSafeCall( cufftSetStream(fft_plan, kernelStream));

		cufftSafeCall(cufftPlan1d( &hilbert_plan, frameWidth, CUFFT_C2C, frameHeight *  framesPerBuffer));
		cufftSafeCall( cufftSetStream(hilbert_plan, kernelStream));
}


extern "C" void initCudaProcVar(	int frameWid, 
									int frameHei, 
									int framesPerBuff,
									float lambMin,
									float lambMax,
									float dispMag,
									float dispValue,
									float dispValueThird,
									int interpMethod,
									int fftLenMult)
{
	frameWidth = frameWid;
	frameHeight = frameHei;
	framesPerBuffer = framesPerBuff;
	lambdaMin = lambMin;
	lambdaMax = lambMax;
	dispMagn = dispMag;
	dispVal = dispValue;
	dispValThird = dispValueThird;
	bufferSize = frameWid*frameHei*framesPerBuff;
	numThreadsPerBlock = 256;
	while (bufferSize/numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	printf("Number of Threads Per Block is: %d \n\n", numThreadsPerBlock);
	samplingMethod = interpMethod;
	fftLengthMult = fftLenMult;

	if (samplingMethod==0 || samplingMethod==1) {
		minVal = MINVAL;
		maxVal = MAXVAL;
	} else if (samplingMethod==2) {
		minVal = MINVAL_WITH_PREFILTER;
		maxVal = MAXVAL_WITH_PREFILTER;
	}
}


extern "C" void cleanUpCUDABuffers()
{
	//Clean up all CUDA Buffers and arryays
	cutilSafeCall(cudaFree(dev_ushortBuffer));
	cutilSafeCall(cudaFree(dev_floatBuffer));
	cutilSafeCall(cudaFree(lookupLambda));
	cutilSafeCall(cudaFree(linearIdx));
	cutilSafeCall(cudaFree(dcArray));
	cutilSafeCall(cudaFree(dispPhaseCartesian));
	cutilSafeCall(cudaFree(dev_FFTCompBuffer));
	cutilSafeCall(cudaFree(dev_CompBuffer));

	//Clean up FFT plans created
	cufftSafeCall(cufftDestroy(fft_plan));
	cufftSafeCall(cufftDestroy(hilbert_plan));

	//Clean up the streams created
	cudaStreamDestroy(memcpyStream);
	cudaStreamDestroy(kernelStream);

	mallocArrays = false;
}


//This Function calls the kernel which averages the given number of frames into a single frame (B-scan)
extern "C" void frameAvg(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int numberOfFrames, int frameNum)
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
extern "C" void copySingleFrame(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int frameNum)
{
	if (dev_multiFrameBuff==NULL) {
		dev_multiFrameBuff = dev_tempBuffer;
	}
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height / dimBlockX.x);
	copySingleFrameFloat<<<dimGridX, dimBlockX, 0, kernelStream>>>
		(dev_multiFrameBuff, dev_displayBuff, frameNum, width*height);
}


extern "C" void cudaPipeline(	unsigned short *h_buffer, 
								float *dev_frameBuff, 
								int frameIdx,
								int reduction, //This is used only for Downsizing
								int offset, //This is used only for Cropping
								int range) //This is used only for Cropping
{
	if (!mallocArrays) {
		initProcCuda();
		mallocArrays = true;
	}

	if (dev_frameBuff==NULL) {
		dev_frameBuff = dev_tempBuffer;
	}

	//initCUDABuffers(h_buffer, host_floatFrameBuffer);
	//Interpolation must be performed before MemcpyAsync to avoid race condition
	//Interpolation will be performed with default stream, therefore prevents all possibilities of race condition
	interp_and_DCSub(dev_CompBuffer);

	//initCUDABuffers(h_buffer, host_floatFrameBuffer);
	//Do a Host to Device Memcpy if using Swept Source, or using Prefiltered Cubic
	//Otherwise, memcpy will happen later directly in Host to Array, instead of Host to Device
	if ((samplingMethod==2)) {
		cutilSafeCall( cudaMemcpyAsync((void *) dev_ushortBuffer, h_buffer, bufferSize*sizeof(unsigned short), cudaMemcpyHostToDevice, memcpyStream));
	} 

//Full Pipeline Implementation
	//For Linear Interpolation and Non-Prefiltered Cubic Interpolation
	else if (samplingMethod==0 || samplingMethod==1) {
		initUshortTexture(h_buffer, frameWidth, frameHeight, framesPerBuffer, memcpyStream);
	}

	dispersionCompensation(dev_CompBuffer, dev_FFTCompBuffer, kernelStream);
	batchFFT(dev_FFTCompBuffer, kernelStream);

	//This kernel must be general for 2D OCT, 3D OCT reduce and crop!
	if (reduction==1) {
		postFFTCrop(dev_FFTCompBuffer, dev_frameBuff, framesPerBuffer, frameIdx, offset, range, kernelStream);
	} else {
		postFFTDownsize(dev_FFTCompBuffer, dev_frameBuff, framesPerBuffer, frameIdx, reduction, kernelStream);
	}


	if (samplingMethod==2) { 
		castKernel<<<bufferSize/numThreadsPerBlock, numThreadsPerBlock>>>(dev_ushortBuffer, dev_floatBuffer);
		callCubicPrefilter(dev_floatBuffer, frameWidth*sizeof(float), frameWidth, frameHeight*framesPerBuffer, numThreadsPerBlock, memcpyStream);
		initFloatTexture(dev_floatBuffer, frameWidth, frameHeight, framesPerBuffer, memcpyStream);
	}

	//This method is used for synchronization with memcpystream as reference
	cudaStreamSynchronize ( memcpyStream );
	//END of Synchronization Calls.
}



extern "C" void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 is the most optimum size for this kernel
	bool partialFundus = false;
	if (dev_volume == NULL) {
		dev_volume = dev_tempBuffer;
		partialFundus = true;
	}
	const int blockSize = 128;
	float scaleCoeff = 4.0f/(float)frameWidth;
	int increment = depth;

	while (height*increment>65535) {
		increment >>= 1;
	}

	dim3 dimBlockX(blockSize);
	dim3 dimGridX(height*increment);

	if (partialFundus) {
			renderFundus<blockSize><<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, 0, height*idx);
	} else {
		for (int i=0; i<depth; i+=increment) {
			renderFundus<blockSize><<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, height*i, height*i);
		}
	}
}


/*****************************************************************************************************************************/
/******************************************  Miscellaneous Functions For Adjustments *****************************************/
/*****************************************************************************************************************************/
float valueIncrement = 0.2f;


extern "C" void changeSamplingMethod(unsigned int sampleMethod)
{
	samplingMethod = sampleMethod;

	if (samplingMethod==0 || samplingMethod==1) {
		minVal = MINVAL;
		maxVal = MAXVAL;
	} else if (samplingMethod==2) {
		minVal = MINVAL_WITH_PREFILTER;
		maxVal = MAXVAL_WITH_PREFILTER;
	}
	dcAcquired = false;
}
/*****************************************************************************************************************************/

extern "C" void acquireDC()
{
	dcAcquired = false;
}
/*****************************************************************************************************************************/
extern "C" void decreaseDispVal()
{
	dispVal -= 0.000001f;
	printf("New DispVal is %0.6f", dispVal);
	printf("\n");
	dispPhaseCalculated = false;
}
/*****************************************************************************************************************************/
extern "C" void increaseDispVal()
{
	dispVal += 0.000001f;
	printf("New DispVal is %0.6f", dispVal);
	printf("\n");
	dispPhaseCalculated = false;
}
/*****************************************************************************************************************************/
extern "C" void decreaseMinVal()
{
	minVal -= valueIncrement;
	printf("New minVal is %0.1f", minVal);
	printf("\n");
}
/*****************************************************************************************************************************/
extern "C" void increaseMinVal()
{
	if (minVal==maxVal-1) {
		printf("Error: minVal cannot be equal or greater than maxVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		minVal += valueIncrement;
		printf("New minVal is %0.1f", minVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
extern "C" void increaseMaxVal()
{
	maxVal += valueIncrement;
	printf("New maxVal is %0.1f", maxVal);
	printf("\n");
}
/*****************************************************************************************************************************/
extern "C" void decreaseMaxVal()
{
	if (maxVal==minVal+1) {
		printf("Error: maxVal cannot be equal or less than than minVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		maxVal -= valueIncrement;
		printf("New maxVal is %0.1f", maxVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
