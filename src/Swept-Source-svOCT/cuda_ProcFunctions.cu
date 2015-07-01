/**********************************************************************************
Filename	: cuda_ProcFunctions.cu
Authors		: Jing Xu, Kevin Wong, Yifan Jian, Marinko Sarunic
Published	: Janurary 6th, 2014

Copyright (C) 2014 Biomedical Optics Research Group - Simon Fraser University
This software contains source code provided by NVIDIA Corporation.

This file is part of a Open Source software. Details of this software has been described 
in the papers titled: 

"Jing Xu, Kevin Wong, Yifan Jian, and Marinko V. Sarunic.
'Real-time acquisition and display of flow contrast with speckle variance OCT using GPU'
In press (JBO)
and
"Jian, Yifan, Kevin Wong, and Marinko V. Sarunic. 'GPU accelerated OCT processing at 
megahertz axial scan rate and high resolution video rate volumetric rendering.' 
In SPIE BiOS, pp. 85710Z-85710Z. International Society for Optics and Photonics, 2013."

Function "dftregistration" is implemented refering to the algorithm in
"Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image 
registration algorithms," Opt. Lett. 33, 156-158 (2008)."

Please refer to these papers for further information about this software. Redistribution 
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
#include <cufft.h> //This is to perform FFT using CUDA
#include "cuda_ProcHeader.cuh" 
#include "windows.h"


typedef float2 Complex;
int numThreadsPerBlock;

bool mallocArrays;
bool dcAcquired;
bool lambdaCalculated;
bool dispPhaseCalculated;
enum bufferType { A, B };
bufferType processBufferID;


int	frameWidth;
int frameHeight;
int framesPerBuffer;
int frameCounter;
int bufferSize;
int fftLengthMult;
float minVal;
float maxVal;
float fundusCoeff;

float *dev_tempBuffer;
unsigned short *dev_uShortBufferA;
unsigned short *dev_uShortBufferB;


cufftHandle fft2d_plan;
cufftHandle fft2d_Batchplan;//plan for framesPerBuffer size
cufftHandle fft2d_BatchplanS;//plan for framesPerBuffer/3 size

float *dcArray;
Complex *dev_FFTCompBuffer;
cufftHandle fft_plan;
cudaStream_t memcpyStream;
cudaStream_t kernelStream;

//for registration
float *MaxV;
int *shift;
int *Nw;
int *Nh;	
float *diffphase;
int *RegLoc;
float *RegMaxV;
float *MaxVB;
int *shiftB;
float *diffphaseB;
int *RegLocB;
float *RegMaxVB;

Complex *sub_FFTCompBufferBatch;
Complex *sub_FFTCompBufferBatchTemp;
Complex *sub_FFTCompBufferBatchSmall;
Complex *sub_FFTCompBufferBatchSmallTemp;

float *sub_absBatchSmall;

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
		//registration section
		cudaMalloc((void**)&sub_FFTCompBufferBatch,frameWidth*frameHeight *framesPerBuffer *sizeof(Complex));
		cudaMalloc((void**)&sub_FFTCompBufferBatchTemp,frameWidth*frameHeight *framesPerBuffer *sizeof(Complex));
		cudaMalloc((void**)&MaxV, sizeof(float));
		cudaMalloc((void**)&shift, 2*sizeof(int));
		cudaMalloc((void**)&RegLoc, frameHeight *sizeof(int));
		cudaMalloc((void**)&RegMaxV, frameHeight *sizeof(float));
		cudaMemset(RegLoc, 0, frameHeight * sizeof(float));
		cudaMemset(RegMaxV, 0, frameHeight * sizeof(float));
		cudaMalloc((void**)&diffphase, sizeof(float));
		cudaMalloc((void**)&Nw, frameWidth*frameHeight*sizeof(int));
		cudaMalloc((void**)&Nh, frameWidth*frameHeight*sizeof(int));
		fft2dPlanCreate(frameHeight, frameWidth,1);
		getMeshgridFunc(frameWidth, frameHeight);

		cudaMalloc((void**)&MaxVB, framesPerBuffer /3*sizeof(float));
		cudaMalloc((void**)&shiftB, framesPerBuffer /3*2*sizeof(int));
		cudaMalloc((void**)&diffphaseB, framesPerBuffer /3*sizeof(float));
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
	frameCounter = 0;
	fundusCoeff = 4.0f;

	numThreadsPerBlock = 256;

	mallocArrays = false;
	dcAcquired = true;
	lambdaCalculated = false;
	dispPhaseCalculated = false;
	processBufferID = A;

	 //65535 is currently the maximum number of threads per kernel
	while (bufferSize/numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, CUDA is unable to handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}

	minVal = 9.5;
	maxVal = 12.5;
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
	copySingleFrameFloat<<<dimGridX, dimBlockX>>>
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
	cudaMemcpyAsync((void *) memcpyBuffer, h_buffer, (bufferSize)*sizeof(unsigned short), cudaMemcpyHostToDevice, memcpyStream);
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


void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx, bool partialFundus,int funOffset, int funWidth,bool MIP)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 is the most optimum size for this kernel

	int inputIdx = height*idx;
	if (dev_volume == NULL) {
		dev_volume = dev_tempBuffer;
		inputIdx = 0;
	}
	const int blockSize = 128;
	float scaleCoeff = fundusCoeff/(float)frameWidth*1000;//The scaling factor was chosen for display purpose
	int increment = depth;

	while (height*increment>65535) {
		increment >>= 1;
	}

	dim3 dimBlockX(blockSize);
	dim3 dimGridX(height*increment);
	if(width<funWidth)
		funWidth = width;

	if(MIP){
		if (partialFundus) {
			renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, inputIdx, height*idx, funOffset,funWidth);
		} else {
			for (int i=0; i<depth; i+=increment) {				
				renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
					(dev_volume, dev_fundus, width, scaleCoeff, height*i, height*i, funOffset, funWidth);
			}
		}
	}
	else{
		if (partialFundus) {
			renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, inputIdx, height*idx, funOffset,funWidth);
		} else {
				// Temporary solution... height*i --> 0  height*i-->height*idx
				renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, 0, height*idx, funOffset,funWidth);
			//}
		}
	}
}

void cudaSvRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx, bool partialFundus,int funOffset, int funWidth,bool MIP)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 provides the suitable performance for the hardware described in the papers.
	int inputIdx = height*idx*3;
	const int blockSize = 128;
	float scaleCoeff = fundusCoeff/(float)frameWidth*1000; //The scaling factor here was chosen for display purpose
	float scaleCoeffMIP = fundusCoeff/(float)frameWidth*5; //The scaling factor here was chosen for display purpose
	int increment = depth;

	while (height*increment>65535) {
		increment >>= 1;
	}

	dim3 dimBlockX(blockSize);
	dim3 dimGridX(height*increment);

		if(width<funWidth)
		funWidth = width;

	if(MIP){
		if (partialFundus) {
				MIPrenderFundusSV<<<dimGridX, dimBlockX, 0, kernelStream>>>
					(dev_volume, dev_fundus, width, scaleCoeffMIP, inputIdx, height*idx, funOffset,funWidth, height);

		} else {
			for (int i=0; i<depth; i+=increment) {
				MIPrenderFundusSV<<<dimGridX, dimBlockX, 0, kernelStream>>>
					(dev_volume, dev_fundus, width, scaleCoeffMIP, height*i, height*i, funOffset, funWidth, height);
			}
		}
	}
	else{
		if (partialFundus) {
			renderFundusSV<<<dimGridX, dimBlockX, 0, kernelStream>>>
				(dev_volume, dev_fundus, width, scaleCoeff, inputIdx, height*idx, funOffset,funWidth, height);
			
		} else {
			for (int i=0; i<depth; i+=increment) {
				renderFundusSV<<<dimGridX, dimBlockX, 0, kernelStream>>>
					(dev_volume, dev_fundus, width, scaleCoeff, height*i, height*i, funOffset, funWidth, height);			
			}
		}
	}
}

/*****************************************************************************************************************************/
/******************************************  Miscellaneous Functions For Adjustments *****************************************/
/*****************************************************************************************************************************/
void acquireDC()
{
	dcAcquired = false;
}
/*****************************************************************************************************************************/
void decreaseMinVal()
{
	minVal -= 0.1f;
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
		minVal += 0.1f;
		printf("New minVal is %0.1f", minVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
void increaseMaxVal()
{
	maxVal += 0.1f;
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
		maxVal -= 0.1f;
		printf("New maxVal is %0.1f", maxVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
void setMinMaxVal(float min,float max,float funCoeff)
{
	maxVal = max;
	minVal = min;
	fundusCoeff = funCoeff;
	printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
}
/*****************************************************************************************************************************/
void increaseFundusCoeff()
{
	fundusCoeff += 1.0f;
	printf("New fundusCoeff is %0.1f", fundusCoeff);
	printf("\n");
}
/*****************************************************************************************************************************/
void decreaseFundusCoeff()
{
	if (fundusCoeff<=0) {
		printf("Error: fundusCoeff cannot be less than 0!\n");
	} else {
		fundusCoeff -= 1.0f;
		printf("New fundusCoeff is %0.1f", fundusCoeff);
		printf("\n");
	}
}

/*****************************************************************************************************************************/
//This Kernel will copy the first frame to the other location of the volume buffer 
//For registration purpose

void speckleVar(float *d_volumeBuffer, float *dev_displayBuff,  float *dev_svFrameBuff, int width, int height, int numF, int frameNum,bool average)
{
	if (d_volumeBuffer==NULL) {
		d_volumeBuffer = dev_tempBuffer;
	}
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height*framesPerBuffer/3/ dimBlockX.x);
	float coefficient = 20; //The scaling factor 20 here was chosen for display purpose

	Variance<<<dimGridX, dimBlockX, 0, kernelStream>>>
		(d_volumeBuffer, dev_displayBuff, dev_svFrameBuff, numF, frameNum, width*height, coefficient);
}
/*****************************************************************************************************************************/

void regisMulB(Complex *src,int *Nw,int *Nh,int width,int height,int frameNum, int framesPerBufferm, float *diffphase, int *shift){

	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height*framesPerBuffer/3 / dimBlockX.x);
	ImagExpB<<<dimGridX, dimBlockX>>>(src, Nw, Nh, width, height,frameNum, framesPerBuffer,shift,diffphase);
}


void fft2dPlanCreate(int height, int width,int subPixelFactor)
{
	cufftPlan2d(&fft2d_plan, height, width,CUFFT_C2C);
	int n[2] = {height, width};
	int n_fac[2] = {height*subPixelFactor, width*subPixelFactor};
	cufftPlanMany(&fft2d_Batchplan, 2, n, 
				  NULL, 1,0, // *inembed, istride, idist 
				  NULL, 1,0, // *onembed, ostride, odist
				  CUFFT_C2C, framesPerBuffer);
	cufftPlanMany(&fft2d_BatchplanS, 2, n_fac, 
				  NULL, 1,0, // *inembed, istride, idist 
				  NULL, 1,0, // *onembed, ostride, odist
				  CUFFT_C2C, framesPerBuffer/3);
}
void fft2dPlanDestroy()
{
	cufftDestroy(fft2d_plan);
	cufftDestroy(fft2d_Batchplan);
	cufftDestroy(fft2d_BatchplanS);
}
void fftBufferCreate(int height, int width, int subPixelFactor)
{
	cudaMalloc((void**)&sub_FFTCompBufferBatchSmall,subPixelFactor*subPixelFactor*width*height*framesPerBuffer /3*sizeof(Complex));
	cudaMalloc((void**)&sub_FFTCompBufferBatchSmallTemp,subPixelFactor*subPixelFactor*width*height*framesPerBuffer /3*sizeof(Complex));
	cudaMalloc((void**)&sub_absBatchSmall,subPixelFactor*subPixelFactor*width*height*framesPerBuffer /3*sizeof(float));
	cudaMemset(sub_FFTCompBufferBatchSmall, 0, subPixelFactor*subPixelFactor*width*height *framesPerBuffer /3*sizeof(Complex));
	cudaMemset(sub_FFTCompBufferBatchSmallTemp, 1, subPixelFactor*subPixelFactor*width*height *framesPerBuffer /3*sizeof(Complex));
	cudaMalloc((void**)&RegLocB, framesPerBuffer /3*height*subPixelFactor *sizeof(int));
	cudaMalloc((void**)&RegMaxVB, framesPerBuffer /3*height*subPixelFactor *sizeof(float));
}
void fftBufferDestroy()
{
	cudaFree(sub_FFTCompBufferBatchSmall);
	cudaFree(sub_FFTCompBufferBatchSmallTemp);
	cudaFree(sub_absBatchSmall);
	cudaFree(RegLocB);
	cudaFree(RegMaxVB);
}
void getMeshgridFunc(int width, int height)
{
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height / dimBlockX.x);
	getMeshgrid<<<dimGridX, dimBlockX>>>(Nw, Nh, width,height);	
}

void dftregistration(	float *Src,
						int subPixelFactor,
						int width,
						int height,
						int numF,
						int frameNum)
{
	syncKernel <<<1,1>>> ();
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height*framesPerBuffer/ dimBlockX.x);
	dim3 dimGridXS(width*height*framesPerBuffer/ dimBlockX.x/3);
	// Update the width and height
	int Nwidth = width*subPixelFactor;
	int Nheight = height*subPixelFactor;
	dim3 dimGridXfac(Nwidth*Nheight*framesPerBuffer/ dimBlockX.x/3);
	const int blockSize = 128;
	dim3 dimBlockX1(blockSize);
	dim3 dimGridX1(Nheight*framesPerBuffer/3);


	copyToComplex <<<dimGridX, dimBlockX,0, kernelStream>>> ( Src, sub_FFTCompBufferBatch);
	
	// sub-pixel registration
	// subPixelFactor ---> Upsampling factor (integer)
	// only subPixelFactor = 1&2 are implemented. >2 will be implemented in a future version.
	// subPixelFactor = 1 ----> whole-pixel shift -- Compute crosscorrelation by an IFFT and locate the peak
	// subPixelFactor = 2 ----> Images will be registered to within 1/2 of a pixel. 
	
	// Section 1: To correct the 1st and 2nd B-scans in a BM-scan
	if (subPixelFactor >1) 
	{
		// subPixelFactor = 2 ----> First upsample by a factor of 2 to obtain initial estimate
		// Embed Fourier data in a 2x larger array
		cufftExecC2C(fft2d_Batchplan,
			(cufftComplex *) sub_FFTCompBufferBatch,
			(cufftComplex *) sub_FFTCompBufferBatch,
			CUFFT_FORWARD);
	
		fftshift2D<<<dimGridX,dimBlockX>>>
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchTemp, width,height);
		
		complexMulConj<<<dimGridXS, dimBlockX>>>
			(sub_FFTCompBufferBatchTemp,sub_FFTCompBufferBatchSmall,0,numF,width,height,subPixelFactor);
		
		fftshift2D<<<dimGridXfac,dimBlockX>>>
			(sub_FFTCompBufferBatchSmall,sub_FFTCompBufferBatchSmallTemp,Nwidth ,Nheight);

		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}
	else
	{
		cufftExecC2C(fft2d_Batchplan,
		(cufftComplex *) sub_FFTCompBufferBatch,
		(cufftComplex *) sub_FFTCompBufferBatch,
		CUFFT_FORWARD);
	
		complexMulConj<<<dimGridXS, dimBlockX>>>
			(sub_FFTCompBufferBatch,sub_FFTCompBufferBatchSmallTemp,0,numF,width,height,subPixelFactor);
	
		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}
	normData<<<dimGridXfac, dimBlockX>>>(sub_FFTCompBufferBatchSmallTemp,1/float(Nwidth*Nheight));
	batchComplexAbs<<<dimGridXfac, dimBlockX>>>(sub_FFTCompBufferBatchSmallTemp,sub_absBatchSmall, 0);

	// Compute crosscorrelation and locate the peak 
	maxReductionBatch<<<dimGridX1, dimBlockX1>>>
		(sub_absBatchSmall, RegMaxVB, Nwidth,Nheight,RegLocB);

	//Obtain shift in original pixel grid from the position of the crosscorrelation peak 
	if (subPixelFactor >1)
	computeShift<<<1,framesPerBuffer/3>>>
		(RegMaxVB,RegLocB,Nwidth,Nheight,0,framesPerBuffer,MaxVB,diffphaseB,sub_FFTCompBufferBatchSmallTemp,shiftB,subPixelFactor);
	else
	computeShift<<<1,framesPerBuffer/3>>>
		(RegMaxVB,RegLocB,Nwidth,Nheight,0,framesPerBuffer,MaxVB,diffphaseB,sub_FFTCompBufferBatch,shiftB,subPixelFactor);
	
	getMeshgridFunc(width, height);   

	//Compute registered version
	regisMulB(sub_FFTCompBufferBatch,Nw,Nh,width,height,1,framesPerBuffer,diffphaseB,shiftB);
	
	// Section 2: To correct the the 3rd scan based on the corrected 2nd B-scans in a BM-scan
	if (subPixelFactor>1)
	{
		fftshift2D<<<dimGridX,dimBlockX>>>
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchTemp,width,height );
		complexMulConj<<<dimGridXS, dimBlockX>>>
			(sub_FFTCompBufferBatchTemp,sub_FFTCompBufferBatchSmall,1,numF,width,height,subPixelFactor);
		fftshift2D<<<dimGridXfac,dimBlockX>>>
			(sub_FFTCompBufferBatchSmall,sub_FFTCompBufferBatchSmallTemp, Nwidth,Nheight);
		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}
	else
	{
		complexMulConj<<<dimGridXS, dimBlockX>>>
			(sub_FFTCompBufferBatch,sub_FFTCompBufferBatchSmallTemp,1,numF,width,height,subPixelFactor);
		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *) sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);	
	}
	normData<<<dimGridXfac, dimBlockX>>>(sub_FFTCompBufferBatchSmallTemp,1/float(Nwidth*Nheight));

	batchComplexAbs<<<dimGridXfac, dimBlockX>>>(sub_FFTCompBufferBatchSmallTemp,sub_absBatchSmall, 0);
	
	// Compute crosscorrelation and locate the peak 
	maxReductionBatch<<<dimGridX1, dimBlockX1>>>
				(sub_absBatchSmall, RegMaxVB, Nwidth,Nheight,RegLocB);

	//Obtain shift in original pixel grid from the position of the crosscorrelation peak 
	if (subPixelFactor>1)
	computeShift<<<1,framesPerBuffer/3>>>
		(RegMaxVB,RegLocB,Nwidth,Nheight,1,framesPerBuffer,MaxVB,diffphaseB,sub_FFTCompBufferBatchSmallTemp,shiftB,subPixelFactor);
	else
	computeShift<<<1,framesPerBuffer/3>>>
		(RegMaxVB,RegLocB,Nwidth,Nheight,1,framesPerBuffer,MaxVB,diffphaseB,sub_FFTCompBufferBatch,shiftB,subPixelFactor);

	//Compute registered version
	regisMulB(sub_FFTCompBufferBatch,Nw,Nh,width,height,2,framesPerBuffer,diffphaseB,shiftB);
	
	cufftExecC2C(fft2d_Batchplan,
				(cufftComplex *)sub_FFTCompBufferBatch,
				(cufftComplex *)sub_FFTCompBufferBatch,
				CUFFT_INVERSE);
	normData<<<dimGridX, dimBlockX>>>(sub_FFTCompBufferBatch,1/float(width*height));
	batchComplexAbs<<<dimGridX, dimBlockX>>>(sub_FFTCompBufferBatch,Src, 0);
}