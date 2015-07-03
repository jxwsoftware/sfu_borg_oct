/**********************************************************************************
Filename	: cuda_ProcKernels.cu
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

#include "cuda_Header.cuh"

texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> texRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefPref; 
cudaArray* cuArray;
enum mallocType {nullType, uShortType, floatType};
mallocType currentMalloc = nullType;


/******** DEVICE FUNCTIONS **********/
__device__ Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__device__ float complexAbs(Complex a)
{
    float c;
	c = sqrt( pow(a.x, 2) + pow(a.y, 2) );
    return c;
}


/******** GLOBAL FUNCTIONS **********/

__global__ void castKernel(unsigned short *d_buff, float *d_fbuff)    
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	d_fbuff[idx] = (float)d_buff[idx];
}


//This is the DC Acquisition Kernel
//Takes the average of many A-scans to obtain a general averaged DC line
__global__ void dcAcquireKernel (Complex *Src, float *Dst, 
                           int width,
                           int imageheight)    
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

    //Sum up all columns of accross Ascans
    for (unsigned int n=0; n<imageheight; n++)
        Dst[idx] += Src[idx + n*width].x;

    Dst[idx] /= (float) imageheight;
}

__global__ void dcSubKernel (Complex *DstSrc, 
                          float *DCval,
                          int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*width + x;

	DstSrc[idx].x = DstSrc[idx].x - DCval[idx%width];
}


__global__ void interp_DCSub(	unsigned int sampMethod, 
								int width, 
								int fftWidth,
								int height,
								float *lambdaCoordinates,
								float *dcArray,
								bool dcAcquired,
								Complex *output)
{ 
    // Calculate normalized texture coordinates 
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//Experimented that shared memory did not help in this case
	int lineNum = idx/width;
	float u = lambdaCoordinates[idx%width] + 0.5f;
    float v = lineNum + 0.5f;
    // Read from texture and write to global memory 

	float tempVal;
	tempVal = tex2D(texRef, u, v);

	//Different output depending on whether DC has been acquired or not
	if (dcAcquired) {
		output[idx].x = tempVal - dcArray[idx%width];
	} else {
		output[idx].x = tempVal;
	}
	output[idx].y = 0;
}



__global__ void hilbertCoeff(	Complex *d_RawData, 
								int width)
{
  //get total number of threads and current thread number
  //blockDim and gridDim are 1 dimensional vectors (y dim = 1) 
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

    //the current thread will run sample_cnt/numThreads times
    if ((idx % width) > (width/2)) // chop out the negative half
    {
        d_RawData[idx].x = 0;
        d_RawData[idx].y = 0;
    }
	else if ((idx % width)== 0 || (idx % width)== (width/2) ) {/* DO NOTHING*/}
    else//upscale the positive half
    {
        d_RawData[idx].x = d_RawData[idx].x * 2;
        d_RawData[idx].y = d_RawData[idx].y * 2;
    }
}


//This kernel is to obtain the Dispersion Phase Vector
__global__ void getDispersion(	float a2, float a3, float dispMag,
								float *d_kLinear, Complex *d_Disp,
								int width)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

    __shared__ float DispPhase;

    // All magnitudes are 10, so no need for another variable
    DispPhase =	a2 * pow((d_kLinear[idx]-1.0),2) 
				+ a3 * pow((d_kLinear[idx]-1.0),3);

    // convert to cartesian coordinates
    d_Disp[idx].x = dispMag * cos(DispPhase);
    d_Disp[idx].y = dispMag * sin(DispPhase);
}


//This Kernel is to multiply the cartesian equivalent of Dispersion Phase with Data
__global__ void dispComp_and_PadComplex(Complex *SrcComplex, 
										Complex *Src1, 
										Complex *DstComplex,
										int width,
										int fftWidth)
{
  //get total number of threads and current thread number
  //blockDim and gridDim are 1 dimensional vectors (y dim = 1)
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	int dstIdx = int(idx/width)*fftWidth + idx%width;

	//No Scaling after Hilbert FFT
	DstComplex[dstIdx] = ComplexMul( SrcComplex[idx], Src1[idx%width]);
	DstComplex[dstIdx+width].x = 0;
	DstComplex[dstIdx+width].y = 0;
}

//DownSizing post fft kernel
__global__ void downsizeMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int reduction)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//Calculating the indices beforehand is much faster than calculating these indices during the function,
	//Therefore it would make sense to do all of it here first before maping the values
	int newWidth = width/reduction;
	int newHeight = height/reduction;
	int newFftWidth = fftWidth/reduction;

	int cmplxIdx = idx%newWidth + int(idx/newWidth)*newFftWidth;
	int buffFrameNum = cmplxIdx/(newFftWidth*newHeight);
	int rowNum = (cmplxIdx%(newFftWidth*newHeight))/newFftWidth;
	int rowIdx = (cmplxIdx%(newFftWidth*newHeight))%newFftWidth;

	int mapFloatIdx = frameIdx*newWidth*newHeight + idx;
	int mapCmpIdx = buffFrameNum*(fftWidth*height) + (rowNum*fftWidth + rowIdx)*reduction;

	floatArray[mapFloatIdx] = 
						__saturatef( (logf( (complexAbs(complexArray[mapCmpIdx])+ 1)) - minVal)*coeff);
}



//Crop Post FFT Method
//ALS = Absolute, Log, and Scaling
//This method of post FFT crops out a certain portion of the data, and copies into buffer
//As opposed to the other method which downsizes the whole volume
__global__ void cropMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int offset, int range)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//Calculating the indices beforehand is much faster than calculating these indices during the function,
	//Therefore it would make sense to do all of it here first before maping the values
	int mapFloatIdx = frameIdx*range*height + idx;
	int mapCmpIdx = int(idx/range)*fftWidth + idx%range + offset;

	floatArray[mapFloatIdx] = 
						__saturatef((logf((complexAbs(complexArray[mapCmpIdx])+ 1)) - minVal)*coeff);
}


__global__ void copySingleFrameFloat(float *Src, float *Dst, int frameNum, int frameSize) 
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	Dst[idx] = Src[frameSize*frameNum + idx];
}


//Frame Average Kernel
__global__ void avgKernel(float *src_Buffer, float *dst_Buffer, int frameNum, int frames, int frameSize)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ float tempVal;
	tempVal = 0;

	for(int i=0; i<frames; i++)
	{
		tempVal += src_Buffer[(frameNum+i)*frameSize + idx];
	}
	dst_Buffer[idx] = tempVal/frames;
}


__device__ void warpReduce(volatile float *sdata, unsigned int tid, unsigned int blockSize) 
{
	if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
	if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
	if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
	if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
	if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
	if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}


__global__ void renderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	int rowIdx = blockIdx.x + inputOffset;
	int outputRowIdx = blockIdx.x + outputOffset;

	sdata[tid] = 0;
	for (int j=0; j<width; j+=blockDim.x)
		sdata[tid] += g_idata[rowIdx*width + tid + j];

	__syncthreads();

	if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	if (tid < 32) warpReduce(sdata, tid, blockDim.x);
	if (tid == 0) g_odata[outputRowIdx] = sdata[0]*scaleCoeff; //Equivalent to 7.0f/1024.0f, multiplication much faster than division
}


/************************************************************************************/
__global__ void syncKernel()
{
	//This Kernel is Purposely Empty
	//By calling a non-streamed empty kernel, the whole system will be synchronized together
	//This Kernel will NOT affect any CPU threads, therefore it should not pose any problems
}
/************************************************************************************/



/*************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
*************************************************************************************************************************/

extern "C" void cleanUpCudaArray()
{
	checkCudaErrors(cudaFreeArray(cuArray));

}

extern "C" void initUshortTexture(unsigned short *host_array, int width, int height, int numFrames, cudaStream_t thisStream)
{
	int size = width*height*numFrames;
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();

	if (currentMalloc == floatType) {
		cudaUnbindTexture(texRefPref);
		checkCudaErrors(cudaFreeArray(cuArray));
		currentMalloc = nullType;
	}
	if (currentMalloc == nullType) {
		checkCudaErrors( cudaMallocArray(&cuArray, &channelDesc, width, height*numFrames));
		currentMalloc = uShortType;
	}
	checkCudaErrors(cudaMemcpyToArrayAsync(cuArray, 0, 0, host_array, size*sizeof(unsigned short), cudaMemcpyHostToDevice, thisStream));	

	texRef.normalized = false;  // access with normalized texture coordinates
	texRef.filterMode = cudaFilterModeLinear;
	checkCudaErrors(cudaBindTextureToArray(texRef, cuArray, channelDesc));
}


extern "C" void initFloatTexture(float *dev_array, int width, int height, int numFrames, cudaStream_t thisStream)
{
	int size = width*height*numFrames;
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	if (currentMalloc == uShortType) {
		cudaUnbindTexture(texRef);
		checkCudaErrors(cudaFreeArray(cuArray));
		currentMalloc = nullType;
	}

	if (currentMalloc == nullType) {
		checkCudaErrors( cudaMallocArray(&cuArray, &channelDesc, width, height*numFrames));
		currentMalloc = floatType;
	}
	checkCudaErrors(cudaMemcpyToArrayAsync(cuArray, 0, 0, dev_array, size*sizeof(float), cudaMemcpyDeviceToDevice, thisStream));

	texRefPref.normalized = false;
	texRefPref.filterMode = cudaFilterModeLinear;
	checkCudaErrors(cudaBindTextureToArray(texRefPref, cuArray, channelDesc));

}








