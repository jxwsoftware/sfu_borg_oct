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

/* NVIDIA's Disclaimer
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

#include <helper_math.h>
#include <cufft.h> 
#include "cuda_ProcHeader.cuh"

texture<float, 2, cudaReadModeElementType> imageTex;
texture<float, 1, cudaReadModeElementType> gaussianTex;

bool mallocImgArray = false;

cudaArray* d_array, *d_tempArray, *d_gaussianArray;

int gSigma1 = 2;
int gSigma2 = 5;
cufftHandle fft2dnotch_plan;
float *temp_image; //square
Complex *temp_notch;
float *NotchFiltMask2d_D;
float *d_tempGaussianFrame;
cudaArray *d_tempGaussArray;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefPref; 
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();


float *d_tempGaussianMask;
float *h_tempGaussianMask;

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range 
    filter and domain filter, the previous one preserves crisp edges and 
    the latter one filters noise. The intensity value at each pixel in 
    an image is replaced by a weighted average of intensity values from 
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as 
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear, 
    the sample just uses a simple pixel by pixel step. 

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may 
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

__device__ float euclideanLen(float a, float b, float d)
{

	float recip = 1/(2 * d * d);
    float mod = (b - a) * (b - a);
    return __expf(-mod*recip);
}

/*
    Because a 2D gaussian mask is symmetric in row and column,
    here only generate a 1D mask, and use the product by row 
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
//use only one block
__global__ void
d_generate_gaussian(float *og, float delta, int radius)
{
    int x = threadIdx.x - radius;
	float recip = 1/(2 * delta * delta);
    og[threadIdx.x] = __expf(-(x * x) * recip);
}        


//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(float *od, float e_d, int w, int h, int r,float factor)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        float sum = 0.0f;
       float factor_o;
        float t = 0.0f;
        float center = tex2D(imageTex, x, y);

        for(int i = -r; i <= r; i++)
        {
            for(int j = -r; j <= r; j++)
            {
                float curPix = tex2D(imageTex, x + j, y + i);
                factor_o = (tex1D(gaussianTex, i + r) * tex1D(gaussianTex, j + r)) *     //domain factor
                    euclideanLen(curPix, center, e_d); //range factor
				    t += factor_o * curPix; 

                sum += factor_o;
            }
        }
		od[y * w + x] = t / sum;
    }
}

void initTexture(int width, int height, void *pImage)
{
    int size = width * height * sizeof(unsigned int);

    // copy image data to array
	if (!mallocImgArray) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaMallocArray  ( &d_array, &channelDesc, width, height); 
		cudaMallocArray  ( &d_tempArray, &channelDesc, width, height );
		mallocImgArray = true;
	}
    cudaMemcpyToArray( d_array, 0, 0, pImage, size, cudaMemcpyDeviceToDevice);
}

void freeFilterTextures()
{
	if (mallocImgArray) {
		cudaFreeArray(d_array);
		cudaFreeArray(d_tempArray);
		cudaFreeArray(d_gaussianArray);
		mallocImgArray = false;
	}
}

void updateGaussian(float delta, float radius)
{
	if (mallocImgArray) {
		cudaFreeArray(d_gaussianArray);
	}
    int size = 2 * radius + 1;

    float* d_gaussian;
    cudaMalloc((void**) &d_gaussian, (2 * radius + 1)* sizeof(float));

    //generate gaussian array
    d_generate_gaussian<<< 1, size>>>(d_gaussian, delta, radius);

    //create cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray( &d_gaussianArray, &channelDesc, size, 1 ); 
    cudaMemcpyToArray( d_gaussianArray, 0, 0, d_gaussian, size * sizeof (float), cudaMemcpyDeviceToDevice);

    // Bind the array to the texture
    cudaBindTextureToArray( gaussianTex, d_gaussianArray, channelDesc);
    cudaFree(d_gaussian);
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/

void bilateralFilter(	float *d_dest, 
							int width, int height,
							float e_d, float radius, int iterations,
							int nthreads, float factor)
{
    // Bind the array to the texture
    cudaBindTextureToArray(imageTex, d_array);

    for(int i=0; i<iterations; i++) 
    {
        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<< gridSize, blockSize>>>
			(d_dest, e_d, width, height, radius,factor);
		
        if (iterations > 1) {
            // copy result back from global memory to array
            cudaMemcpyToArray( d_tempArray, 0, 0, d_dest, width * height * sizeof(float),cudaMemcpyDeviceToDevice);        
			cudaBindTextureToArray(imageTex, d_tempArray);
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

void Gaussian2DGen(float *h_Mask,int width, int height, float b_w, float b_h, float c_w,float c_h)
{
	float scaleFactor = 0.0;
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			h_Mask[i*height+j] = exp((-1)*((pow(i-b_w,2))/(2*pow(c_w,2))+(pow(j-b_h,2))/(2*pow(c_h,2))));
			scaleFactor += h_Mask[i*height+j];
		}
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			h_Mask[i*height+j] = h_Mask[i*height+j]/scaleFactor; 
}

void initGaussian(int frameWidth, int frameHeight,int gaussFiltWindSize)
{
	cudaMalloc(&d_tempGaussianFrame, frameWidth*frameHeight*sizeof(float));
	cudaMemset(d_tempGaussianFrame, 0, frameWidth*frameHeight*sizeof(float));
	cudaMallocArray  ( &d_tempGaussArray, &channelDesc, frameWidth, frameHeight/3); 

	//Gaussian filter for the fundus view
	int width = gaussFiltWindSize;
	int height = gaussFiltWindSize;
	cudaMalloc(&d_tempGaussianMask, width*height*sizeof(float));
	h_tempGaussianMask = (float *) malloc(width*height*sizeof(float));
	Gaussian2DGen(h_tempGaussianMask,width, height, width/2, height/2, width,height);
	cudaMemset(d_tempGaussianMask, 0, width*height*sizeof(float));
	cudaMemcpy(d_tempGaussianMask,h_tempGaussianMask, width*height*sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void gaussFilter(float *d_srcFrame, float *d_dstFrame,float *d_Mask, int frameWidth, int frameHeight,int gaussFiltWindSize)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int yIdx = idx/frameWidth;
	int xIdx = idx%frameWidth;

	int width = gaussFiltWindSize/2;
	if (yIdx>width && yIdx<frameHeight-width && xIdx>width && xIdx<frameWidth-width)
	{
		d_dstFrame[idx]=0;
		for(int i=0;i<gaussFiltWindSize;i++)
			for (int j = 0;j<gaussFiltWindSize;j++)
			d_dstFrame[idx] += d_srcFrame[(yIdx-(j-width))*frameWidth +xIdx-(i-width)]*d_Mask[i*gaussFiltWindSize+j];
	} 
	else 
		d_dstFrame[idx] =	d_srcFrame[idx];
}

//Gaussian mask for fundus
void gaussianFilterGPU(float *d_image, int imageW, int imageH, int gaussFiltWindSize)
{
	int newWidth = imageW;
	int newHeight = imageH;
	gaussFilter<<<newHeight, newWidth>>>(d_image, d_tempGaussianFrame,d_tempGaussianMask,newWidth, newHeight, gaussFiltWindSize);
	cudaMemcpy(d_image, d_tempGaussianFrame, newWidth*newHeight*sizeof(float), cudaMemcpyDeviceToDevice);
}
void cleanUpFiltBuffers()
{
	cufftDestroy(fft2dnotch_plan);
	cudaFree(temp_notch);
}
__global__ void Gaussian2DGenNotch(float *mask, int width, int height, float a, float b, float c1,float c2)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	
	int xIdx = idx/width;
	int yIdx = idx%width;

	float mask_W = exp((-1)*(pow(xIdx-b,2))/(2*pow(c2,2)));//low pass
	float mask_H = abs(exp((-1)*(pow(yIdx-a,2))/(2*pow(c1,2)))-1);//high pass

	if (mask_W+mask_H >1 )
		mask[idx] = 1;
	else 
		mask[idx] = mask_W+mask_H;
}

__global__ void notchFilter(Complex *d_Frame, float* mask, int cenW, int offsetW_1, int cenH, int offsetH_1)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int yIdx = idx/300;
	int xIdx = idx%300;
	
	if (xIdx >= cenW - offsetW_1 && xIdx < cenW +offsetW_1)
	{
		if (yIdx < cenH -offsetH_1 && yIdx >= cenH +offsetH_1)
		{
			d_Frame[idx].x = mask[xIdx]*d_Frame[idx].x;
			d_Frame[idx].y = mask[xIdx]*d_Frame[idx].y;
		}
	}
	else
	{
		d_Frame[idx].x = mask[xIdx]*d_Frame[idx].x;
		d_Frame[idx].y = mask[xIdx]*d_Frame[idx].y;
	}
}

__global__ void notchFilter2d(Complex *d_Frame, float* mask)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	
	d_Frame[idx].x = mask[idx]*d_Frame[idx].x;
	d_Frame[idx].y = mask[idx]*d_Frame[idx].y;
}
__global__ void complexToAbsN(Complex *a, float* d_image) 
{	
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
    d_image[idx] = sqrt( pow(a[idx].x, 2) + pow(a[idx].y, 2));
}
__global__ void fftshift_2D(Complex *data, int N1, int N2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    int i = idx%N2;
    int j = idx/N2;

    if (i < N1 && j < N2) {
       float a = 1-2*((i+j)&1);

       data[idx].x *= a;
       data[idx].y *= a; 
    }
}
void decreaseNotchGaussianSigmaH(int imageW, int imageH)
{
	if (gSigma2 < 1) {
		printf("Error: gSigma2 cannot be less than 0!\n");
	} else {
		gSigma2 -= 1;
		printf("New gSigma2 for horizontal is %d", gSigma2);
		printf("\n");
	}
	if (imageW>imageH)// at this point, the width is always larger than height
		imageH = imageW;
	else
		imageH = imageW;
	Gaussian2DGenNotch<<<imageH*imageW/256,256>>>(NotchFiltMask2d_D,imageW,imageH,imageW/2,imageH/2,gSigma1,gSigma2);
}
void increaseNotchGaussianSigmaH(int imageW, int imageH)
{		gSigma2 += 1;
		printf("New gSigma2 for horizontal is %d", gSigma2);
		printf("\n");
	if (imageW>imageH)// at this point, the width is always larger than height
		imageH = imageW;
	else
		imageH = imageW;
	Gaussian2DGenNotch<<<imageH*imageW/256,256>>>(NotchFiltMask2d_D,imageW,imageH,imageW/2,imageH/2,gSigma1,gSigma2);
}
void decreaseNotchGaussianSigmaV(int imageW, int imageH)
{
	if (gSigma1 < 1) {
		printf("Error: gSigma1 cannot be less than 0!\n");
	} else {
		gSigma1 -= 1;
		printf("New gSigma1 for vertical is %d", gSigma1);
		printf("\n");
	}
	if (imageW>imageH)// at this point, the width is always larger than height
		imageH = imageW;
	else
		imageH = imageW;
	Gaussian2DGenNotch<<<imageH*imageW/256,256>>>(NotchFiltMask2d_D,imageW,imageH,imageW/2,imageH/2,gSigma1,gSigma2);
}
void increaseNotchGaussianSigmaV(int imageW, int imageH)
{
		gSigma1 += 1;
		printf("New gSigma1 for vertical %d", gSigma1);
		printf("\n");
	if (imageW>imageH)// at this point, the width is always larger than height
		imageH = imageW;
	else
		imageH = imageW;
	Gaussian2DGenNotch<<<imageH*imageW/256,256>>>(NotchFiltMask2d_D,imageW,imageH,imageW/2,imageH/2,gSigma1,gSigma2);
}
__global__ void padImage(float *src, float *dst, int width,int height)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
	int rIdx = idx/width%height;
	int cIdx = idx%width;

	dst[idx] = src[rIdx*width+cIdx];
}
__global__ void unpadImage(float *src, float *dst, int width,int height)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	dst[idx] = src[idx];
}
void notchFilterGPU(
    float *d_image,
    int imageW,
    int imageH
)
{
	//	The dimension of the image has to set to be the same for this function
	int imageW_N = imageW;
	int imageH_N = imageH;

	if (imageW>imageH)// at this point, the width is always larger than height
		imageH_N = imageW;
	else
		imageW_N = imageH;
	dim3 dimBlockX(256);
	dim3 dimGridX(imageW_N*imageH_N/ dimBlockX.x);	
	dim3 dimGridXs(imageW*imageH/ dimBlockX.x);

	cudaMemset(temp_image,0,imageH_N*imageW_N*sizeof(float));
	cudaMemset(temp_notch,0,imageH_N*imageW_N*sizeof(Complex));

	if (imageW == imageH)
		copyToComplex <<<dimGridX, dimBlockX>>> (d_image, temp_notch);
	else{
		padImage<<<dimGridX, dimBlockX>>>(d_image,temp_image,imageW,imageH);
		copyToComplex <<<dimGridX, dimBlockX>>> (temp_image, temp_notch);
	}
	fftshift_2D<<<dimGridX, dimBlockX>>>(temp_notch, imageH_N,imageW_N);	
	cufftExecC2C(fft2dnotch_plan,
		(cufftComplex *) temp_notch,
		(cufftComplex *) temp_notch,
		CUFFT_FORWARD);
	
	notchFilter2d<<<dimGridX,dimBlockX>>>(temp_notch,NotchFiltMask2d_D);

	cufftExecC2C(fft2dnotch_plan,
		(cufftComplex *) temp_notch,
		(cufftComplex *) temp_notch,
		CUFFT_INVERSE);
	normData<<<dimGridX, dimBlockX>>>(temp_notch,1/float(imageW_N*imageH_N));
	if (imageW == imageH)
		complexToAbsN <<<dimGridX, dimBlockX>>> (temp_notch,d_image);
	else{
		complexToAbsN <<<dimGridX, dimBlockX>>> (temp_notch,temp_image);
		unpadImage<<<dimGridXs, dimBlockX>>>(temp_image,d_image,imageW,imageH);	
	}
}
// 
void initNotchFiltVarAndPtrs(int imageH, int imageW)
{
//image with same dimension
	if (imageW>imageH)
		imageH = imageW;
	else
		imageW = imageH;
	cufftPlan2d(&fft2dnotch_plan, imageW,imageH, CUFFT_C2C);
	cudaMalloc((void**)&temp_notch,imageH*imageW*sizeof(Complex));
	cudaMalloc((void**) &temp_image, imageH*imageW* sizeof(float));
	cudaMalloc((float**)&NotchFiltMask2d_D,imageH*imageW*sizeof(float));
	cudaMemset(NotchFiltMask2d_D,0,imageH*imageW*sizeof(float));
	//Generate the mask for the notch filter
	Gaussian2DGenNotch<<<imageW*imageH/256,256>>>(NotchFiltMask2d_D,imageW,imageH,imageW/2,imageH/2,gSigma1,gSigma2);	
}
#endif
