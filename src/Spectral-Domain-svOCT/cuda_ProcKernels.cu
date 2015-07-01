/**********************************************************************************
Filename	: cuda_FilterKernels.cu
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

/********	A note on this file:
/********	cuda_ProcKernels.cu has already been included in cuda_ProcFunctions.cu, and should NOT be compiled as part of the "Source Files"
/********	This file should either be "excluded from build" or excluded from the project completely, and should NOT be compiled.
********/

#include "cuda_ProcHeader.cuh" 

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


//This Kernel multiplies the cartesian equivalent of Dispersion Phase with Data
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
						__saturatef( (log10f( (complexAbs(complexArray[mapCmpIdx])+ 1)) - minVal)*coeff);
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
		__saturatef((log10f((complexAbs(complexArray[mapCmpIdx])+ 1)) - minVal)*coeff);
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


__global__ void renderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset, int funoff, int funwid) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];


	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	int rowIdx = blockIdx.x + inputOffset;
	int outputRowIdx = blockIdx.x + outputOffset;

	sdata[tid] = 0;


	for (int j=0; (tid+j)<funwid; j+=blockDim.x)
		sdata[tid] += g_idata[rowIdx*width + tid + j+funoff];


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

__global__ void getPhase(Complex *d_fftData, float *dst_phase,int width, 
							 int height, int fftWidth,
							 int frameIdx, int offset, int range)   
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int mapFloatIdx = frameIdx*range*height + idx;
	int mapCmpIdx = int(idx/range)*fftWidth + idx%range + offset;
	dst_phase[mapFloatIdx] = atan2f(d_fftData[mapCmpIdx].y,d_fftData[mapCmpIdx].x);
}

/*************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
**************************************************************************************************************************
*************************************************************************************************************************/

void cleanUpCudaArray()
{
	cudaFreeArray(cuArray);

}

void initUshortTexture(unsigned short *host_array, int width, int height, int numFrames, cudaStream_t thisStream)
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


void initFloatTexture(float *dev_array, int width, int height, int numFrames, cudaStream_t thisStream)
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


		dim3 dimBlockY(min(PowTwoDivider(width), yThreads));
		dim3 dimGridY(width / dimBlockY.x);
		SamplesToCoefficients2DY<floatN><<<dimGridY, dimBlockY, 0, processStream>>>(image, pitch, width, height);

 }

 //This function will call the function above
void callCubicPrefilter(float *dev_Coeffs, int pitch, int width, int height, int threadPerBlock, cudaStream_t processStream)
{
	int xKernelThreads = 16;
	int yKernelThreads = 256;
	return CubicBSplinePrefilter2DStreamed(dev_Coeffs, pitch, width, height, xKernelThreads, yKernelThreads, processStream);
}
// End of Cubic Prefilter Calls
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Variance(float *src_Buffer, float *dst_Buffer, float *dst_svBuffer, int numF, int frameNum, int frameSize)
{
  
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx/(frameSize); //0-9
	int inFrameIdx = idx%(frameSize);

	float tempVal;
	tempVal = 0;

	for (int i=0;i<numF;i++)
	 tempVal += src_Buffer[(zIdx*numF + frameNum+i)*frameSize + inFrameIdx]; //zIdx*numF = 0:3:6:9:12...:27
	 
	float mean = tempVal/numF;
	float var = 0;
	for (int i=0;i<numF;i++)
	var += pow(src_Buffer[(zIdx*numF + frameNum+i)*frameSize + inFrameIdx]-mean,2);
	tempVal = var/numF*100; //The scaling factor 20 here was chosen for display purpose
	src_Buffer[(zIdx*numF + frameNum)*frameSize + inFrameIdx] = tempVal;
}
__global__ void copyMulFrameFloat(float *Src, float *Dst, int frameNum, int frameSize, int numF) 
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	for (int i=0;i<numF;i++)
	Dst[frameSize*(frameNum+i) + idx] = Src[idx];
}
//registration section//
__global__ void complexMulConj(Complex *Src, Complex *Dst, int frameNum, int frames, int width, int height, int subPixelFactor)
{
	int frameSize = width*height;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx/(frameSize); 
	int inFrameIdx = idx%(frameSize);
	Complex temp;
	int a = (zIdx*frames+frameNum)*frameSize;
	int b = (zIdx*frames+frameNum+1)*frameSize;
	temp.x = Src[a + inFrameIdx].x * Src[b+inFrameIdx].x - Src[a+inFrameIdx].y * (-1)*Src[b+inFrameIdx].y;
    temp.y = Src[a + inFrameIdx].x * Src[b+inFrameIdx].y*(-1) + Src[a + inFrameIdx].y * Src[b+inFrameIdx].x;
	
	int outFrameIdx = 0;
	if (subPixelFactor == 1)
		outFrameIdx = idx;
	else
		outFrameIdx = (inFrameIdx/width+height/2)* width*subPixelFactor +(inFrameIdx%width+width/2) +zIdx*frameSize*4;	
	Dst[outFrameIdx] = temp;	
}
__global__ void batchComplexAbs( Complex *Src, float *Dst, int offset) 
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	Dst[offset + idx] = complexAbs(Src[idx]);
}

__global__ void initFloat(float *input, float val) 
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	input[idx] = sinf((float)idx);
} 
__global__ void copyToComplex( float *input, Complex *output)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	output[idx].x = input[idx];
	output[idx].y = 0.0f;
}
__global__ void normData(Complex *input, float norm) 
{	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    input[idx].x *= norm;
    input[idx].y *= norm;
 }

__device__ void MaxWarpReduce(volatile float *sdata, unsigned int tid, int blockSize,int* loc) 
{
	if (blockSize >=  64) {
		if (sdata[tid] < sdata[tid +32]){
			sdata[tid] = sdata[tid +32];
			loc[tid] = loc[tid +32];
		}
	}
	if (blockSize >=  32) {
		if (sdata[tid] < sdata[tid +16]){
			sdata[tid] = sdata[tid +16];
			loc[tid] = loc[tid +16];
		}
	}
	if (blockSize >=  16) {
		if (sdata[tid] < sdata[tid +8]){
			sdata[tid] = sdata[tid +8];
			loc[tid] = loc[tid +8];
		}
	}
	if (blockSize >=  8){
		if (sdata[tid] < sdata[tid +4]){
			sdata[tid] = sdata[tid +4];
			loc[tid] = loc[tid +4];
		}
	}
	if (blockSize >=  4) {
		if (sdata[tid] < sdata[tid +2]){
			sdata[tid] = sdata[tid +2];
			loc[tid] = loc[tid +2];
		}
	}
	if (blockSize >=  2) {
		if (sdata[tid] < sdata[tid +1]){
			sdata[tid] = sdata[tid +1];
			loc[tid] = loc[tid +1];
		}
	}
}
__global__ void maxReduction(float *g_idata, float *maxV, unsigned int width,  int* loc) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];
	__shared__ int sloc[1024];
	int blockSize = blockDim.x;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = 0;
	for (int j=tid; j<width; j+=blockSize){
		if (sdata[tid] < g_idata[blockIdx.x*width + j]){
			sdata[tid] = g_idata[blockIdx.x*width + j];
			sloc[tid] = j;
		}
	}	
		__syncthreads();

	if (blockSize >= 128) { 
		if (tid <  64) {
			if (sdata[tid] < sdata[tid +  64]){
			sdata[tid] = sdata[tid +  64];
			sloc[tid] = sloc[tid + 64];
			}
		}
		__syncthreads();

		if (tid < 32) MaxWarpReduce(sdata, tid, blockSize,sloc);

		if (tid == 0) {
			maxV[blockIdx.x] = sdata[0];
			loc[blockIdx.x] = sloc[tid];
		}
	}
	
}
__global__ void maxReductionBatch(float *g_idata, float *maxV, unsigned int width,int height,  int* loc) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];
	__shared__ int sloc[1024];
	int blockSize = blockDim.x;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = 0;
	for (int j=tid; j<width; j+=blockSize){
		if (sdata[tid] < g_idata[(blockIdx.x)*width + j]){
			sdata[tid] = g_idata[(blockIdx.x)*width + j];
			sloc[tid] = j;
		}
	}	
		__syncthreads();

	if (blockSize >= 128) { 
		if (tid <  64) {
			if (sdata[tid] < sdata[tid +  64]){
			sdata[tid] = sdata[tid +  64];
			sloc[tid] = sloc[tid + 64];
			}
		}
		__syncthreads();

		if (tid < 32) MaxWarpReduce(sdata, tid, blockSize,sloc);

		if (tid == 0) {
			maxV[blockIdx.x] = sdata[tid];
			loc[blockIdx.x] = sloc[tid];
		}
	}
}
__global__ void computeShift(float *RegMaxV, int *RegLoc, int width,
						int height,int offsetFrame,int framesPerBuffer, float *MaxV,float *diffphase, Complex *data, int *shift,int subPixelFactor) 
{	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx*height;
	MaxV[idx] = RegMaxV[zIdx];
	int hloc;
	int wloc;
	hloc = 0;
	wloc = RegLoc[zIdx];
	for (int j=1; j<height; j++){
		if (MaxV[idx] < RegMaxV[zIdx+j]){
			MaxV[idx] = RegMaxV[zIdx+j];
			hloc = j;
			wloc = RegLoc[zIdx+j];
		}
	}	
	int md2 = width/2;
	int nd2 = height/2;
	if (wloc > md2)
		shift[idx] = wloc - width + 1;
	else		
		shift[idx] =  wloc; 
	if (hloc > nd2)
		shift[idx+framesPerBuffer/3] = hloc - height +1; 
	else
		shift[idx+framesPerBuffer/3] = hloc;
	shift[idx] /=subPixelFactor;
	shift[idx+framesPerBuffer/3] /= subPixelFactor;
	// diffphase ---> Global phase difference between the two images (should be zero if images are non-negative).
	// diffphase[idx] = atan2(data[(idx*3 + offsetFrame)*width/subPixelFactor*height/subPixelFactor+ hloc/subPixelFactor*width/subPixelFactor +wloc/subPixelFactor].y,data[(idx*3 + offsetFrame)*width/subPixelFactor*height/subPixelFactor+hloc/subPixelFactor*width/subPixelFactor +wloc/subPixelFactor].x);	
	// For our OCT processing pipeline, the intensity of processed images are only from 0-1.
	diffphase[idx] = 0;
}
__global__ void getMeshgrid( int *Nw, int *Nh, int width,int height)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int dstIdx1 = int(idx/width);
	int dstIdx2 = idx%width;
	if (dstIdx2 < (width/2))
		Nw[idx] = dstIdx2;
	else
		Nw[idx] = dstIdx2 - width;
	
	if (dstIdx1 < (height/2))
		Nh[idx] = dstIdx1;
	else
		Nh[idx] = dstIdx1 - height;
}
__global__ void ImagExp (Complex *Src, int *Nw, int *Nh, float width, float height, int *shift,float *diffphase)
{
	float theta;
	Complex r;
	Complex s;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	theta = 2*_PI*((-1)*(float(shift[0])*float(Nw[idx])/width + float(shift[1])*float(Nh[idx])/height));
	
	r.x = cosf(theta);
	r.y = sinf(theta);
	s.x = cosf(diffphase[0]);
	s.y = sinf(diffphase[0]);
	Src[idx] = ComplexMul(Src[idx],ComplexMul(r,s));
}
__global__ void ImagExpB (Complex *Src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBuffer,int *shift,float *diffphase)
{
	float theta;
	Complex r;
	Complex s;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx/(width*height);
	int InframeIdx = idx%(width*height);
	theta = 2*_PI*((-1)*(float(shift[zIdx])*float(Nw[InframeIdx])/width + float(shift[zIdx+framesPerBuffer/3])*float(Nh[InframeIdx])/height));
	r.x = cosf(theta);
	r.y = sinf(theta);
	s.x = cosf(diffphase[zIdx]);
	s.y = sinf(diffphase[zIdx]);	
	Src[(zIdx*3+frameNum)*width*height+InframeIdx] = ComplexMul(Src[(zIdx*3+frameNum)*width*height+InframeIdx],ComplexMul(r,s));
}
__device__ void MIPwarpReduce(volatile float *sdata, unsigned int tid, int blockSize) 
{
	if (blockSize >=  64) {
		if (sdata[tid] < sdata[tid +32]){
			sdata[tid] = sdata[tid +32];
			
		}
	}
	if (blockSize >=  32) {
		if (sdata[tid] < sdata[tid +16]){
			sdata[tid] = sdata[tid +16];
		
		}
	}
	if (blockSize >=  16) {
		if (sdata[tid] < sdata[tid +8]){
			sdata[tid] = sdata[tid +8];
			
		}
	}
	if (blockSize >=  8){
		if (sdata[tid] < sdata[tid +4]){
			sdata[tid] = sdata[tid +4];
			
		}
	}
	if (blockSize >=  4) {
		if (sdata[tid] < sdata[tid +2]){
			sdata[tid] = sdata[tid +2];
			
		}
	}
	if (blockSize >=  2) {
		if (sdata[tid] < sdata[tid +1]){
			sdata[tid] = sdata[tid +1];
			
		}
	}
}
__global__ void MIPrenderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset, int funoff, int funwid) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];
	int rowIdx = blockIdx.x + inputOffset;
	int outputRowIdx = blockIdx.x + outputOffset;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = 0;
	for (int j=0; (tid+j)<funwid; j+=blockDim.x){
		if (sdata[tid] < g_idata[rowIdx*width + tid + j+funoff]){
			sdata[tid] = g_idata[rowIdx*width + tid + j+funoff];
		
		}
	}	
		__syncthreads();

	if (blockDim.x >= 128) { 
		if (tid <  64) {
			if (sdata[tid] < sdata[tid +  64]){
			sdata[tid] = sdata[tid +  64];
		
			}
		}
		__syncthreads();

		if (tid < 32) MIPwarpReduce(sdata, tid, blockDim.x);

		if (tid == 0) {
			g_odata[outputRowIdx]  = sdata[0]*scaleCoeff;

		}
	}
	
}

__global__ void MIPrenderFundusSV(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset, int funoff, int funwid, int height) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];

	int svZIdx = (blockIdx.x/height) * 3;
	int svInFrameIdx =  blockIdx.x%height;
	int svBlockIdx = svZIdx*height + svInFrameIdx;
	int rowIdx = svBlockIdx + inputOffset;
	int outputRowIdx = blockIdx.x + outputOffset;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = 0;
	for (int j=0; (tid+j)<funwid; j+=blockDim.x){
		if (sdata[tid] < g_idata[rowIdx*width + tid + j+funoff]){
			sdata[tid] = g_idata[rowIdx*width + tid + j+funoff];
		}
	}	
		__syncthreads();

	if (blockDim.x >= 128) { 
		if (tid <  64) {
			if (sdata[tid] < sdata[tid +  64]){
			sdata[tid] = sdata[tid +  64];
		
			}
		}
		__syncthreads();

		if (tid < 32) MIPwarpReduce(sdata, tid, blockDim.x);

		if (tid == 0) {
			g_odata[outputRowIdx]  = sdata[0]*scaleCoeff;

		}
	}
	
}


__global__ void renderFundusSV(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset, int funoff, int funwid, int height) 
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];
	int blockSize = blockDim.x;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

		int svZIdx = (blockIdx.x/height) * 3;
	int svInFrameIdx =  blockIdx.x%height;
	int svBlockIdx = svZIdx*height + svInFrameIdx;
	int rowIdx = svBlockIdx + inputOffset;
	int outputRowIdx = blockIdx.x + outputOffset;

	sdata[tid] = 0;
	
	for (int j=0; (tid+j)<funwid; j+=blockDim.x)
	sdata[tid] += g_idata[rowIdx*width + tid + j+funoff];

	sdata[tid]=sdata[tid]/width;
	__syncthreads();

	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	if (tid < 32) warpReduce(sdata, tid, blockSize);
	if (tid == 0) g_odata[outputRowIdx] = sdata[0]*scaleCoeff; //Equivalent to 7.0f/1024.0f, multiplication much faster than division
}

__global__ void setColor(float *g_odata,float *g_idata,int index)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	g_odata[idx*3+index] = g_idata[idx];
}

__global__ void fftshift2D( Complex *input, Complex *output, int width, int height)
{
 int frameSize = width*height;
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 int zIdx = idx/(frameSize); //0-9
 int inFrameIdx = idx%(frameSize);
 int x1 = inFrameIdx/width;
 int y1 = inFrameIdx%width;
 int outIdx = ((y1+width/2)%width) + ((x1+height/2)%height)*width+ zIdx*frameSize;
 
 output[outIdx] = input[idx];
}







