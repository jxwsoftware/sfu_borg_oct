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
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_
#include <cutil_inline.h>    //includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>


bool mallocVolumeArray = false;
typedef unsigned int  uint;
typedef unsigned char uchar;
cudaArray *d_volumeArray = 0;
cudaStream_t renderStream;
texture<float, 3, cudaReadModeElementType> tex;         // 3D texture

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}


__global__ void
d_render(float *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, float voxelThreshold)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color

	__shared__ float sum[256];
	__shared__ float subtractValue[256];
	__shared__ float opacThreshold[256];
    float t = tnear;

	int thrIdx = threadIdx.x;
	sum[thrIdx] = 0;
	subtractValue[thrIdx] = 0;
	opacThreshold[thrIdx] = 0.90f;

    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for(int i=0; i<maxSteps; i++) {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);

		sample *= 0.2f;

		if (sum[thrIdx]>0.0f) {
			subtractValue[thrIdx] += 0.01f;
			opacThreshold[thrIdx] -= 0.02f;
		}
		if (sum[thrIdx]==0.0f && sample > voxelThreshold) {
			sum[thrIdx] += sample;
		} else if (sum[threadIdx.x]>0.0f && sample - subtractValue[thrIdx] > 0.0f) {
			sum[thrIdx] += sample - subtractValue[thrIdx];
		}

		if (sum[thrIdx] >= opacThreshold[thrIdx]) break;


        t += tstep;
        if (t > tfar) break;

        pos += step;
    }
	d_output[y*imageW + x] = sum[thrIdx];
}

/*************************************************************************************************************************/
/***************************************  END OF KERNELS   ***************************************************************/
/*************************************************************************************************************************/

//Initialization for MemcpyDeviceToDevice, for Processing AND Volume Rendering
extern "C" void initRayCastCuda(void *d_volume, cudaExtent volumeSize, cudaMemcpyKind memcpyKind)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	if (!mallocVolumeArray) {
		cudaStreamCreate(&renderStream);
		cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );
		mallocVolumeArray = true;
	}
    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(d_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = memcpyKind;
    cutilSafeCall( cudaMemcpy3D(&copyParams) );  

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    cutilSafeCall(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

extern "C" void freeVolumeBuffers()
{
    cutilSafeCall(cudaFreeArray(d_volumeArray));
	mallocVolumeArray = false;
}

extern "C"
void rayCast_kernel(dim3 gridSize, dim3 blockSize, float *d_output, int imageW, int imageH, 
				   float density, float brightness, float transferOffset, float transferScale,
				   float voxelThreshold)
{
	d_render<<<gridSize, blockSize, 0, renderStream>>>( d_output, imageW, imageH, density, 
										brightness, transferOffset, transferScale, voxelThreshold);
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix) );
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_

