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
#include <cutil_inline.h> //This is to perform CUDA safecall functions
#include <cuda_runtime.h>

typedef float2 Complex;




/******** DEVICE FUNCTIONS **********/

__device__ Complex ComplexMul(Complex srcA, Complex srcB)
{
    Complex output;
    output.x = srcA.x * srcB.x - srcA.y * srcB.y;
    output.y = srcA.x * srcB.y + srcA.y * srcB.x;
    return output;
}

__device__ float complexAbs(Complex input)
{
    float output;
	output = sqrt( pow(input.x, 2) + pow(input.y, 2) );
    return output;
}


/******** GLOBAL FUNCTIONS **********/
////This Kernel is to multiply the cartesian equivalent of Dispersion Phase with Data
__global__ void subDC_PadComplex(	unsigned short *Src, 
									Complex *DstComplex,
									float *dcArray,
									int width,
									int fftWidth)
{

  //get total number of threads and current thread number
  //blockDim and gridDim are 1 dimensional vectors (y dim = 1)
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	int dstIdx = int(idx/width)*fftWidth + idx%width;
	int dcIdx = idx%width;

	// This 'short' cast is NOT Necessary
	// In fact some data may not work with the short cast
	DstComplex[dstIdx].x = (float)(short)Src[idx] - dcArray[dcIdx];
	//DstComplex[dstIdx].x = (float)Src[idx] - dcArray[dcIdx];
	DstComplex[dstIdx].y = 0;

	//The following loop is to pad any extra padding with zeros
	//The if-statement must be added to avoid unecessary for-loop reads
		//if width=fftwidth, such as the case when padding is not required
	//In fact, the following for loop can be commented out for the
		//case where padding is not required, which can help save a bit of time
	//The advantage of having the following is that it becomes dynamic
		//when dealing with fftwidth>width
	if (fftWidth>width) {
		int newDstIdx = dstIdx+width;
		DstComplex[newDstIdx].x = 0;
		DstComplex[newDstIdx].y = 0;
	}
}


//This is the DC Acquisition Kernel
//Takes the average of many A-scans to obtain a general averaged DC line
__global__ void dcAcquireKernel ( unsigned short *Src, float *Dst, 
								int width,
								int imageheight)    
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	Dst[idx] = 0;
    //Sum up all columns of accross Ascans
    for (unsigned int n=0; n<imageheight; n++)
        Dst[idx] += (float)(short)Src[idx + n*width];
    Dst[idx] /= (float)imageheight;
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


__global__ void avgKernel(float *src_Buffer, float *dst_Buffer, int frameNum, int frames, int frameSize)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ float tempVal;
	tempVal = 0;

	for(int i=0; i<frames; i++)
		tempVal += src_Buffer[(frameNum+i)*frameSize + idx];
	dst_Buffer[idx] = tempVal/frames;
}



template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) 
{
	if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
	if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
	if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
	if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
	if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
	if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}


template <unsigned int blockSize>
__global__ void renderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int offset) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];

	//Each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	int rowIdx = blockIdx.x + offset;

	//Unroll the initial values to sum into the first 128 elements
	sdata[tid] = 0;
	for (int j=0; j<width; j+=blockSize)
		sdata[tid] += g_idata[rowIdx*width + tid + j];
	__syncthreads();

	//Reduction of the remaining elements
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	//Assign the zeroth index value, containing the sum, into global memory
	if (tid == 0) g_odata[rowIdx] = sdata[0]*scaleCoeff;
} 	

__global__ void syncKernel()
{
	//This Kernel is Purposely Empty
	//By calling a non-streamed empty kernel, the whole system will be synchronized together
	//This Kernel will NOT affect any CPU threads, therefore it should not pose any problems
}



