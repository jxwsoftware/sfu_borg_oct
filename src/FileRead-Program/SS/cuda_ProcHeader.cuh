/**********************************************************************************
Filename	: cuda_ProcFunctions.cu
Authors		: Jing Xu, Kevin Wong, Yifan Jian, Marinko Sarunic
Published	: Janurary 6th, 2014

Copyright (C) 2014 Biomedical Optics Research Group - Simon Fraser University
This software contains source code provided by NVIDIA Corporation.

This file is part of a free software. Details of this software has been described 
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

#include <stdlib.h>
#include <stdio.h>
#include <helper_math.h>

typedef float2 Complex;
#define _PI 3.14159265358979


/*************************************************************************************************************************/
/*******************************************    Cuda Processing Pipeline     *********************************************/
void acquireDC();
void decreaseMinVal();
void increaseMinVal();
void increaseMaxVal();
void decreaseMaxVal();
void setMinMaxVal(float min,float max,float funCoeff);
void increaseFundusCoeff();
void decreaseFundusCoeff();
void cleanUpCUDABuffers();
void cudaPipeline( unsigned short *h_buffer, float *dev_frameBuff, int frameIdx, int reduction, int offset, int range);

void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth);
void frameAvg(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int numberOfFrames, int frameNum);
void copySingleFrame(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int frameNum);

/******** GLOBAL KERNEL FUNCTIONS **********/
__global__ void subDC_PadComplex(	unsigned short *Src, 
									Complex *DstComplex,
									float *dcArray,
									int width,
									int fftWidth);
__global__ void dcAcquireKernel ( unsigned short *Src, float *Dst, 
								int width,
								int imageheight);
__global__ void downsizeMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int reduction);
__global__ void cropMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int offset, int range);
__global__ void copySingleFrameFloat(float *Src, float *Dst, int frameNum, int frameSize);
__global__ void avgKernel(float *src_Buffer, float *dst_Buffer, int frameNum, int frames, int frameSize);
__global__ void avgKernel2(float *src_Buffer, float *dst_Buffer, int frameNum, int frames, int frameSize);
__global__ void renderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int offset);
__global__ void syncKernel();
__global__ void normData(Complex *input, float norm);
__global__ void copyToComplex( float *input, Complex *output);



/*****************************************    Cuda Volume Rendering Pipeline     *****************************************/
/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
/*************************************************************************************************************************/
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void freeVolumeBuffers();
void initRayCastCuda(void *d_volume, cudaExtent volumeSize, cudaMemcpyKind memcpyKind);
void rayCast_kernel(dim3 gridSize, dim3 blockSize, float *d_output, int imageW, int imageH, 
				   float density, float brightness, float transferOffset, float transferScale, float voxelThreshold);

/*************************************************************************************************************************/
/*****************************************    Cuda Bilateral Filtering   *************************************************/
/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
/*************************************************************************************************************************/

void initTexture(int width, int height, void *pImage);
void freeFilterTextures();
void updateGaussian(float delta, int radius);
void bilateralFilter(	float *d_dest,
							int width, int height,
							float e_d, int radius, int iterations,
							int nthreads);
									
/*************************************************************************************************************************/
/*****************************************    Open GL Extern Functions   *************************************************/
/************************************* These functions are declared in this file *****************************************/
void fundusSwitch();
void volumeSwitch();

/*************************************************************************************************************************/


/*************************************************************************************************************************/
/*************************************************************************************************************************/
/***************************************           Subpixel Registration             *************************************/
__global__ void complexMulConj(Complex *Src, Complex *Dst, int frameNum, int frames, int width, int height, int subPixelFactor);
__global__ void batchComplexAbs( Complex *Src, float *Dst, int offset);
__global__ void getMeshgrid( int *Nw, int *Nh, int width,int height);
void fft2dPlanCreate(int height, int width,int subPixelFactor);
void fft2dPlanDestroy();
void fftBufferCreate(int height, int width,int subPixelFactor);
void fftBufferDestroy();
void dftregistration(float *Src, int usfac,int width,int height,int numF,int frameNum);
void getMeshgridFunc(int width, int height);
void regisMulB(Complex *src,int *Nw,int *Nh,int width,int height,int frameNum, int framesPerBufferm,float *diffphase, int *shift);
__global__ void maxReductionBatch(float *g_idata, float *maxV, unsigned int width,int height,  int* loc);
__global__ void computeShift(float *RegMaxV, int *RegLoc, int width,
						int height,int offsetFrame,int framesPerBuffer,float *MaxV,float *diffphase, Complex *data, int *shift,int subPixelFactor);
__global__ void ImagExpB (Complex *Src, int *Nw, int *Nh, int width, int height,int frameNum, int framesPerBuffer,int *shift,float *diffphase);	
__global__ void fftshift2D(Complex *input, Complex *output,  int width, int height);	
/*************************************************************************************************************************/

__global__ void setColor(float *g_odata,float *g_idata,int index);	