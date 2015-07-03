/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Changes in this code has been made to filter only gray levels
 * and NOT RGB. 
 */

#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_math.h>

texture<float, 2, cudaReadModeElementType> imageTex;
texture<float, 1, cudaReadModeElementType> gaussianTex;

bool mallocImgArray = false;

cudaArray* d_array, *d_tempArray, *d_gaussianArray;

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
    Because a 2D gaussian mask is symmetry in row and column,
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
d_bilateral_filter(float *od, float e_d, int w, int h, int r)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;


    if (x < w && y < h) {
        float sum = 0.0f;
        float factor;
        float t = 0.0f;
        float center = tex2D(imageTex, x, y);

        for(int i = -r; i <= r; i++)
        {
            for(int j = -r; j <= r; j++)
            {
                float curPix = tex2D(imageTex, x + j, y + i);
                factor = (tex1D(gaussianTex, i + r) * tex1D(gaussianTex, j + r)) *     //domain factor
                    euclideanLen(curPix, center, e_d); //range factor
                t += factor * curPix;
                sum += factor;
            }
        }
		od[y * w + x] = t / sum;
    }
}


extern "C" 
void initTexture(int width, int height, void *pImage)
{
    int size = width * height * sizeof(unsigned int);

    // copy image data to array
	if (!mallocImgArray) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		checkCudaErrors( cudaMallocArray  ( &d_array, &channelDesc, width, height )); 
		checkCudaErrors( cudaMallocArray  ( &d_tempArray, &channelDesc, width, height ));
		mallocImgArray = true;
	}
    checkCudaErrors( cudaMemcpyToArray( d_array, 0, 0, pImage, size, cudaMemcpyDeviceToDevice));
}

extern "C"
void freeFilterTextures()
{
	if (mallocImgArray) {
		checkCudaErrors(cudaFreeArray(d_array));
		checkCudaErrors(cudaFreeArray(d_tempArray));
		checkCudaErrors(cudaFreeArray(d_gaussianArray));
		mallocImgArray = false;
	}
}


extern "C"
void updateGaussian(float delta, int radius)
{
	if (mallocImgArray) {
		checkCudaErrors(cudaFreeArray(d_gaussianArray));
	}
    int size = 2 * radius + 1;

    float* d_gaussian;
    checkCudaErrors(cudaMalloc( (void**) &d_gaussian, 
        (2 * radius + 1)* sizeof(float)));

    //generate gaussian array
    d_generate_gaussian<<< 1, size>>>(d_gaussian, delta, radius);

    //create cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors( cudaMallocArray( &d_gaussianArray, &channelDesc, size, 1 )); 
    checkCudaErrors( cudaMemcpyToArray( d_gaussianArray, 0, 0, d_gaussian, size * sizeof (float), cudaMemcpyDeviceToDevice));

    // Bind the array to the texture
    checkCudaErrors( cudaBindTextureToArray( gaussianTex, d_gaussianArray, channelDesc));
    checkCudaErrors( cudaFree(d_gaussian) );
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

extern "C" 
void bilateralFilter(	float *d_dest,
							int width, int height,
							float e_d, int radius, int iterations,
							int nthreads)
{
    // Bind the array to the texture
    checkCudaErrors( cudaBindTextureToArray(imageTex, d_array) );

    for(int i=0; i<iterations; i++) 
    {
        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<< gridSize, blockSize>>>
			(d_dest, e_d, width, height, radius);

        if (iterations > 1) {
            // copy result back from global memory to array
            checkCudaErrors( cudaMemcpyToArray( d_tempArray, 0, 0, d_dest, width * height * sizeof(float),
                cudaMemcpyDeviceToDevice));
            checkCudaErrors( cudaBindTextureToArray(imageTex, d_tempArray) );
        }
    }
}

#endif
