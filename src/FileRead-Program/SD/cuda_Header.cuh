
#if defined(__CUDACC__)
typedef float2 Complex;



__global__ void interp_DCSub(	unsigned int sampMethod, 
								int width, 
								int fftWidth,
								int height,
								float *lambdaCoordinates,
								float *dcArray,
								bool dcAcquired,
								Complex *output);
__global__ void dcAcquireKernel (Complex *Src, float *Dst, 
                           int width,
                           int imageheight);
__global__ void dcSubKernel (Complex *DstSrc, 
                          float *DCval,
                          int width);
__global__ void hilbertCoeff(	Complex *d_RawData, 
								int width);
__global__ void getDispersion(	float a2, float a3, float dispMag,
								float *d_kLinear, Complex *d_Disp,
								int width);
__global__ void dispComp_and_PadComplex(Complex *SrcComplex, 
										Complex *Src1, 
										Complex *DstComplex,
										int width,
										int fftWidth);
__global__ void downsizeMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int reduction);
__global__ void cropMLS(float *floatArray, Complex *complexArray, int width, 
							 int height, int fftWidth, float minVal, float maxVal, 
							 float coeff, int frameIdx, int offset, int range);
__global__ void avgKernel(float *src_Buffer, float *dst_Buffer, int frameNum, int frames, int frameSize);
__global__ void copySingleFrameFloat(float *Src, float *Dst, int frameNum, int frameSize);
__global__ void castKernel(unsigned short *d_buff, float *d_fbuff);
__global__ void renderFundus(float *g_idata, float *g_odata, unsigned int width, float scaleCoeff, int inputOffset, int outputOffset);

/*************************************************************************************************************************/
extern "C" void callCubicPrefilter(float *dev_Coeffs, int pitch, int width, int height, int threadPerBlock, cudaStream_t processStream);
extern "C" void initUshortTexture(unsigned short *host_array, int width, int height, int numFrames, cudaStream_t thisStream);
extern "C" void initFloatTexture(float *dev_array, int width, int height, int numFrames, cudaStream_t thisStream);
extern "C" void printComplexArray(Complex *devArray);
extern "C" void printFloatArray(float *devArray);
extern "C" void printUShortArray(unsigned short *devArray);
extern "C" void cleanUpCudaArray();

/*************************************************************************************************************************/






/*************************************************************************************************************************/
/*******************************************    Cuda Processing Pipeline     *********************************************/
extern "C" void changeSamplingMethod(unsigned int sampleMethod);
extern "C" void acquireDC();
extern "C" void decreaseDispVal();
extern "C" void increaseDispVal();
extern "C" void decreaseMinVal();
extern "C" void increaseMinVal();
extern "C" void increaseMaxVal();
extern "C" void decreaseMaxVal();
extern "C" void cleanUpCUDABuffers();
extern "C" void cudaPipeline( unsigned short *h_buffer, float *dev_frameBuff, int frameIdx, int reduction, int offset, int range);
extern "C" void cudaRenderFundus( float *dev_fundus, float *dev_volume, int width, int height, int depth, int idx);
extern "C" void frameAvg(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int numberOfFrames, int frameNum);
extern "C" void copySingleFrame(float *dev_multiFrameBuff, float *dev_displayBuff, int width, int height, int frameNum);

/*************************************************************************************************************************/
/*****************************************    Cuda Volume Rendering Pipeline     *****************************************/
/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
/*************************************************************************************************************************/
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void freeVolumeBuffers();
extern "C" void initRayCastCuda(void *d_volume, cudaExtent volumeSize, cudaMemcpyKind memcpyKind);
extern "C" void rayCast_kernel(dim3 gridSize, dim3 blockSize, float *d_output, int imageW, int imageH, 
				   float density, float brightness, float transferOffset, float transferScale, float voxelThreshold);


/*************************************************************************************************************************/
/*****************************************    Cuda Bilateral Filtering   *************************************************/
/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
/*************************************************************************************************************************/
extern "C" void initTexture(int width, int height, void *pImage);
extern "C" void freeFilterTextures();
extern "C" void updateGaussian(float delta, int radius);
extern "C" void bilateralFilter(float *d_dest, int width, int height,float e_d, 
									int radius, int iterations,int nthreads);
/*************************************************************************************************************************/
/*************************************************************************************************************************/

#endif






/*************************************************************************************************************************/
extern "C" void initGLVarAndPtrs(bool procesData,
								 bool volumeRend,
								 bool fundRend,
								 int frameWid, 
								 int frameHei, 
								 int framesPerBuff,
								 int fileLength,
								 int winWid,
								 int winHei,
								 int interpMethod,
								 int volumeMode);

extern "C" void initCudaProcVar(	int frameWid, 
									int frameHei, 
									int framesPerBuff,
									float lambMin,
									float lambMax,
									float dispMag,
									float dispValue,
									float dispValueThird,
									int interpMethod,
									int fftLenMult);

extern "C" void setBufferPtr( unsigned short *h_buffer);
extern "C" void registerCudaHost();
extern "C" void initGLEvent(int argc, char** argv);
extern "C" void runGLEvent();
/*************************************************************************************************************************/


