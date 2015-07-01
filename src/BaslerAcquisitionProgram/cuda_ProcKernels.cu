/**********************************************************************************
Filename	: cuda_ProcKernels.cu
Authors		: Kevin Wong, Yifan Jian, Jing Xu, Marinko Sarunic
Published	: March 14th, 2013

Copyright (C) 2012 Biomedical Optics Research Group - Simon Fraser University
This software contains source code provided by NVIDIA Corporation.

This file is part of a free software. Details of this software has been described 
in the paper titled: 

"Jian Y, Wong K, Sarunic MV; Graphics processing unit accelerated optical coherence 
tomography processing at megahertz axial scan rate and high resolution video rate 
volumetric rendering. J. Biomed. Opt. 0001;18(2):026002-026002.  
doi:10.1117/1.JBO.18.2.026002."

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


/********	A note on this file:
/********	cuda_ProcKernels.cu has already been included in cuda_ProcFunctions.cu, and should NOT be compiled as part of the "Source Files"
/********	This file should either be "excluded from build" or excluded from the project completely, and should NOT be compiled.
********/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h> //Include the general CUDA Header file
#include <cufft.h> //This is to perform FFT using CUDA
#include <cuda_runtime.h>

//The Functions used for Cubic Spline Interpolation
//These functions are developped by Daniel Ruijters et al.
//These files can be obtained at: http://www.dannyruijters.nl/cubicinterpolation/
//Ensure the parent directory "CubicBSpline" is changed to whichever parent directory they have been installed in
//The name "CubicBSpline" is custom-renamed directory, therefore it will not be the default name after download
#include <CubicBSpline/cubicTex2D.cu>
#include <CubicBSpline/cubicPrefilter2D.cu>
//End of Cubic B-Spline include files

typedef float2 Complex;
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

	//Linear Interpolation
	if (sampMethod == 0) {
		tempVal = tex2D(texRef, u, v);
	}
	//Cubic B-Spline Interpolation
	else if (sampMethod == 1) { 
		tempVal = cubicTex2D(texRef, u, v);
	}

	else if (sampMethod == 2) {
		tempVal = cubicTex2D(texRefPref, u, v);
	}

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
	for (int j=0; j<width; j+=blockSize)
		sdata[tid] += g_idata[rowIdx*width + tid + j];

	__syncthreads();

	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
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
	cudaFreeArray(cuArray);
}

extern "C" void initUshortTexture(unsigned short *host_array, int width, int height, int numFrames, cudaStream_t thisStream)
{
	int size = width*height*numFrames;
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();

	if (currentMalloc == floatType) {
		cudaUnbindTexture(texRefPref);
		cudaFreeArray(cuArray);
		currentMalloc = nullType;
	}
	if (currentMalloc == nullType) {
		cudaMallocArray(&cuArray, &channelDesc, width, height*numFrames);
		currentMalloc = uShortType;
	}
	cudaMemcpyToArrayAsync(cuArray, 0, 0, host_array, size*sizeof(unsigned short), cudaMemcpyHostToDevice, thisStream);	

	texRef.normalized = false;  // access with normalized texture coordinates
	texRef.filterMode = cudaFilterModeLinear;
	cudaBindTextureToArray(texRef, cuArray, channelDesc);
}


extern "C" void initFloatTexture(float *dev_array, int width, int height, int numFrames, cudaStream_t thisStream)
{
	int size = width*height*numFrames;
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	if (currentMalloc == uShortType) {
		cudaUnbindTexture(texRef);
		cudaFreeArray(cuArray);
		currentMalloc = nullType;
	}

	if (currentMalloc == nullType) {
		cudaMallocArray(&cuArray, &channelDesc, width, height*numFrames);
		currentMalloc = floatType;
	}
	cudaMemcpyToArrayAsync(cuArray, 0, 0, dev_array, size*sizeof(float), cudaMemcpyDeviceToDevice, thisStream);

	texRefPref.normalized = false;
	texRefPref.filterMode = cudaFilterModeLinear;
	cudaBindTextureToArray(texRefPref, cuArray, channelDesc);

}


//The following is a modified version of the CubicBSplinePrefilter2D extern function defined in cubicPrefilter2D.cu
//This file can be obtained at: http://www.dannyruijters.nl/cubicinterpolation/
 template<class floatN>
 extern void CubicBSplinePrefilter2DStreamed(floatN* image, uint pitch, uint width, uint height, int xThreads, int yThreads, cudaStream_t processStream)
 {
		dim3 dimBlockX(min(PowTwoDivider(height), xThreads));
		dim3 dimGridX(height / dimBlockX.x);
		SamplesToCoefficients2DX<floatN><<<dimGridX, dimBlockX, 0, processStream>>>(image, pitch, width, height);
		CUT_CHECK_ERROR("SamplesToCoefficients2DX kernel failed");

		dim3 dimBlockY(min(PowTwoDivider(width), yThreads));
		dim3 dimGridY(width / dimBlockY.x);
		SamplesToCoefficients2DY<floatN><<<dimGridY, dimBlockY, 0, processStream>>>(image, pitch, width, height);
		CUT_CHECK_ERROR("SamplesToCoefficients2DY kernel failed");
 }

 //This function will call the function above
extern "C" void callCubicPrefilter(float *dev_Coeffs, int pitch, int width, int height, int threadPerBlock, cudaStream_t processStream)
{
	int xKernelThreads = 16;
	int yKernelThreads = 256;
	return CubicBSplinePrefilter2DStreamed(dev_Coeffs, pitch, width, height, xKernelThreads, yKernelThreads, processStream);
}
// End of Cubic Prefilter Calls
///////////////////////////////////////////////////////////////////////////////////////////////////////////







