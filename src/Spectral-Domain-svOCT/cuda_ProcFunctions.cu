/**********************************************************************************
Filename	: cuda_ProcKernels.cu
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

#include "cuda_ProcHeader.cuh" 

const float MINVAL = 1.8f;
const float MAXVAL = 4.3f;
const float MINVAL_WITH_PREFILTER = 7.70f;
const float MAXVAL_WITH_PREFILTER = 8.70f;
const float INIT_FUNDUS_COEFF = 1.0f;

typedef float2 Complex;
int numThreadsPerBlock = 256;

bool mallocArrays = false;
bool dcAcquired = true;
bool lambdaCalculated = false;
bool dispPhaseCalculated = false;
enum bufferType { A, B };
bufferType processBufferID = A;


float *d_floatFrameBuffer;

int	frameWidth;
int frameHeight;
int framesPerBuffer;
int frameCounter;
int bufferSize;
int fftLengthMult;
float minVal = MINVAL_WITH_PREFILTER;
float maxVal = MAXVAL_WITH_PREFILTER;
float fundusCoeff = INIT_FUNDUS_COEFF;

float dispMagn;
float dispVal;
float dispValThird;
float lambdaMin;
float lambdaMax;

// 0 for linear, 1 for cubic spline, 2 for prefiltered spline
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

cufftHandle fft2d_plan;
cufftHandle fft2d_Batchplan;//plan for framesPerBuffer size
cufftHandle fft2d_BatchplanS;//plan for framesPerBuffer/3 size

cudaStream_t memcpyStream;
cudaStream_t kernelStream;
cudaEvent_t start_event, stop_event;

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

		cudaMemcpy((void *) linearIdx, h_linearIdx, frameWidth*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void *) lookupLambda, h_lookupLambda, frameWidth*sizeof(float), cudaMemcpyHostToDevice);

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

    cufftExecC2C(hilbert_plan,
                 (cufftComplex *)d_ComplexArray,
                 (cufftComplex *)d_ComplexArray,
                 CUFFT_FORWARD);

	hilbertCoeff<<<dimGridX, dimBlockX, 0, processStream>>>
		(d_ComplexArray, frameWidth);

    cufftExecC2C(hilbert_plan,
                 (cufftComplex *)d_ComplexArray,
                 (cufftComplex *)d_ComplexArray,
                 CUFFT_INVERSE);
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
	cufftExecC2C(fft_plan,
				(cufftComplex *)d_ComplexArray,
				(cufftComplex *)d_ComplexArray,
				CUFFT_FORWARD);
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
		cudaStreamCreate(&memcpyStream);
		cudaStreamCreate(&kernelStream);
		cudaMalloc((void**)&dev_tempBuffer, bufferSize * sizeof(float));
		cudaMalloc((void**)&dev_ushortBuffer, bufferSize * sizeof(unsigned short));
		cudaMemset( dev_ushortBuffer, 0, bufferSize * sizeof(unsigned short));
		cudaMalloc((void**)&dev_floatBuffer, bufferSize * sizeof(float));
		cudaMemset( dev_floatBuffer, 0, bufferSize * sizeof(float));

		cudaMalloc((void**)&dcArray, frameWidth * sizeof(float));
		cudaMalloc((void**)&lookupLambda, frameWidth * sizeof(float));
		cudaMalloc((void**)&linearIdx, frameWidth * sizeof(float));
		cudaMalloc((void**)&dispPhaseCartesian, frameWidth * sizeof(Complex));
		cudaMemset(dcArray, 0, frameWidth * sizeof(float));
		cudaMemset(lookupLambda, 0, frameWidth * sizeof(float));
		cudaMemset(linearIdx, 0, frameWidth * sizeof(float));
		cudaMemset(dispPhaseCartesian, 0, frameWidth * sizeof(float));

		cudaMalloc((void**)&dev_CompBuffer, bufferSize * sizeof(Complex));
		cudaMalloc((void**)&dev_FFTCompBuffer, bufferSize * fftLengthMult * sizeof(Complex));

		//Be sure to have the fft_width size be dynamic
		cufftPlan1d( &fft_plan, fftLengthMult*frameWidth, CUFFT_C2C, frameHeight *  framesPerBuffer);
		cufftSetStream(fft_plan, kernelStream);

		cufftPlan1d( &hilbert_plan, frameWidth, CUFFT_C2C, frameHeight *  framesPerBuffer);
		cufftSetStream(hilbert_plan, kernelStream);

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
	//printf("Number of Threads Per Block is: %d \n\n", numThreadsPerBlock);
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


void cleanUpCUDABuffers()
{
	//Clean up all CUDA Buffers and arryays
	cudaFree(dev_ushortBuffer);
	cudaFree(dev_floatBuffer);
	cudaFree(lookupLambda);
	cudaFree(linearIdx);
	cudaFree(dcArray);
	cudaFree(dispPhaseCartesian);
	cudaFree(dev_FFTCompBuffer);
	cudaFree(dev_CompBuffer);
	cudaFree(dev_FFTCompBuffer);
	cudaFree(MaxV);
	cudaFree(shift);
	cudaFree(RegLoc);
	cudaFree(RegMaxV);
	cudaFree(diffphase);
	cudaFree(Nw);
	cudaFree(Nh);	

	//Clean up FFT plans created
	cufftDestroy(fft_plan);
	cufftDestroy(hilbert_plan);
	cufftDestroy(fft2d_plan);

	//Clean up the streams created
	cudaStreamDestroy(memcpyStream);
	cudaStreamDestroy(kernelStream);

	mallocArrays = false;
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

//////////////////////////////////alpha/////////////////////////////////////////////////////////
void readbackPrefix(float *readback_buffer,float *dev_fundus, int height, int depth)
{
	cudaMemcpy( readback_buffer, dev_fundus, height*depth*sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaPipeline(	unsigned short *h_buffer,
					float *dev_frameBuff, 
					float *dev_phaseBuff,
					int frameIdx,
					int reduction, //This is used only for Downsizing
					int offset, //This is used only for Cropping
					int range,
					bool processPhase) //This is used only for Cropping
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
		cudaMemcpyAsync((void *) dev_ushortBuffer, h_buffer, bufferSize*sizeof(unsigned short), cudaMemcpyHostToDevice, memcpyStream);
	} 

//Full Pipeline Implementation
	//For Linear Interpolation and Non-Prefiltered Cubic Interpolation
	else if (samplingMethod==0 || samplingMethod==1) {
		initUshortTexture(h_buffer, frameWidth, frameHeight, framesPerBuffer, memcpyStream);
	}

	dispersionCompensation(dev_CompBuffer, dev_FFTCompBuffer, kernelStream);
	batchFFT(dev_FFTCompBuffer, kernelStream);

	if (processPhase == true) {
		getPhase<<<bufferSize/numThreadsPerBlock,numThreadsPerBlock, 0, kernelStream>>>
			(dev_FFTCompBuffer,dev_phaseBuff,frameWidth,frameHeight,
					 fftLengthMult*frameWidth,frameIdx,offset,range);
	}

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



void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx, bool partialFundus, int funOffset, int funWidth,bool MIP)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 is the most optimum size for this kernel

	int inputIdx = height*idx;
	if (dev_volume == NULL) {
		dev_volume = dev_tempBuffer;
		inputIdx = 0;
	}
	const int blockSize = 128;
	float scaleCoeff = fundusCoeff/(float)frameWidth*10;
	float scaleCoeffMIP = fundusCoeff/(float)frameWidth*60;
	int increment = depth;

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
					(dev_volume, dev_fundus, width, scaleCoeff, inputIdx, height*idx,funOffset, funWidth);
		} else {
			for (int i=0; i<depth; i+=increment) {
				renderFundus<<<dimGridX, dimBlockX, 0, kernelStream>>>
					(dev_volume, dev_fundus, width, scaleCoeff, height*i, height*i,funOffset, funWidth);
			}
		}
	}
}
void cudaSvRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx, bool partialFundus, int funOffset, int funWidth,bool MIP)
{
	//Can be up to 1024, but incredibly inefficient at 1024
	//128 provides the suitable performance for the hardware described in the papers.
	int inputIdx = height*idx*3;
	const int blockSize = 128;
	float scaleCoeff = fundusCoeff/(float)frameWidth*1000;
	float scaleCoeffMIP = fundusCoeff/(float)frameWidth*10;
		//   /(float)frameWidth*100;
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
float valueIncrement = 0.05f;

void changeSamplingMethod(unsigned int sampleMethod)
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

void acquireDC()
{
	dcAcquired = false;
}
/*****************************************************************************************************************************/
void decreaseDispVal()
{
	dispVal -= 0.000001f;
	printf("New DispVal is %0.6f", dispVal);
	printf("\n");
	dispPhaseCalculated = false;
}
/*****************************************************************************************************************************/
void increaseDispVal()
{
	dispVal += 0.000001f;
	printf("New DispVal is %0.6f", dispVal);
	printf("\n");
	dispPhaseCalculated = false;
}
/*****************************************************************************************************************************/
void decreaseMinVal()
{
	minVal -= valueIncrement;
	printf("New minVal is %0.2f", minVal);
	printf("\n");
}
/*****************************************************************************************************************************/
void increaseMinVal()
{
	if (minVal==maxVal-1) {
		printf("Error: minVal cannot be equal or greater than maxVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		minVal += valueIncrement;
		printf("New minVal is %0.2f", minVal);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
void increaseMaxVal()
{
	maxVal += valueIncrement;
	printf("New maxVal is %0.2f", maxVal);
	printf("\n");
}
/*****************************************************************************************************************************/
void decreaseMaxVal()
{
	if (maxVal==minVal+1) {
		printf("Error: maxVal cannot be equal or less than than minVal!\n");
		printf("minVal is: %f, maxVal is: %f \n", minVal, maxVal);
	} else {
		maxVal -= valueIncrement;
		printf("New maxVal is %0.2f", maxVal);
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
	fundusCoeff += 0.5f;
	printf("New fundus Coeff is %0.1f", fundusCoeff);
	printf("\n");
}
/*****************************************************************************************************************************/
void decreaseFundusCoeff()
{
	if (fundusCoeff<=0) {
		printf("Error: fundusCoeff cannot be less than 0!\n");
	} else {
		fundusCoeff -= 0.5f;
		printf("New fundusCoeff is %0.1f", fundusCoeff);
		printf("\n");
	}
}
/*****************************************************************************************************************************/
void resetAllToInitialValues(int sampMethod)
{
	samplingMethod = sampMethod;
	minVal = MINVAL;
	maxVal = MAXVAL;
	fundusCoeff = INIT_FUNDUS_COEFF;
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
	Variance<<<dimGridX, dimBlockX, 0, kernelStream>>>
		(d_volumeBuffer, dev_displayBuff, dev_svFrameBuff, numF, frameNum, width*height);
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
	// only subPixelFactor = 1&2 are implemented. >2 will be implemented in an updated version.
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

	regisMulB(sub_FFTCompBufferBatch,Nw,Nh,width,height,2,framesPerBuffer,diffphaseB,shiftB);
	
	cufftExecC2C(fft2d_Batchplan,
				(cufftComplex *)sub_FFTCompBufferBatch,
				(cufftComplex *)sub_FFTCompBufferBatch,
				CUFFT_INVERSE);
	normData<<<dimGridX, dimBlockX>>>(sub_FFTCompBufferBatch,1/float(width*height));
	batchComplexAbs<<<dimGridX, dimBlockX>>>(sub_FFTCompBufferBatch,Src, 0);
}
