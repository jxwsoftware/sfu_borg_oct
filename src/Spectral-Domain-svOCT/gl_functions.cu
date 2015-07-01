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
#include "gl_Header.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <GL/glew.h> //Required to Generate GL Buffers
#include <GL/freeglut.h>
#include <cuda_gl_interop.h> 

#include "cuda_ProcHeader.cuh" 


//Delay upon which timerevent is called
#define	REFRESH_DELAY	0 //ms

enum volumeDisplay {DownSize, Crop};
volumeDisplay displayMode = Crop;


//Do not modify these initialization values unless you are modifying the program,
//The only values that should be modified during use are in the main, or GUI

//Boolean Variables to determine display and processing functionality
bool processRawData; //Uninitilized value, ensure it is initialized in initGLVarAndPtrs
bool volumeRender; //Uninitilized value, ensure it is initialized in initGLVarAndPtrs
bool fundusRender; //Uninitilized value, ensure it is initialized in initGLVarAndPtrs
bool frameAveraging; //Uninitilized value, ensure it is initialized in initGLVarAndPtrs
bool bilatFilt; //Uninitilized value, ensure it is initialized in initGLVarAndPtrs
bool gaussFilt; //For svOCT 
bool notchFilt; //For svOCT 
bool volumeProcess;
bool slowBscan;
bool autoBscan;
bool SensorlessAO;
bool Cscanselector;
bool displayLine;
bool processPhase = false;
bool speckleVariance = false;
bool moCorr = false;
bool moCorrBool = false;
bool updateRegVar = false;
bool MIP= false;

//Integer variables to determine display window sizes and resolution
int windowWidth;
int windowHeight;
int subWindWidth;
int subWindHeight;
int	width;
int height;
int frames;
int fundusFrames;
int bscanWidth;
int bscanHeight;
int volumeWidth;
int volumeHeight;
int fundusWidth;
int fundusOffset;
int startWidth;
int subPixelFactor;
float bilFilFactor;
int fundusWidth2;
int fundusOffset2;
int fundusWidth3;
int fundusOffset3;


//Processing and Bscan Display Parameters
int framesPerBuffr;
int framesToAvg = 3;
int framesToSvVar = 3;
int frameCount = 0;
int bScanFrameCount = 0;
float tempRatio2 = 0;
int sampMethod = 0;
int svStartFrame = 0;

int gaussFiltWindSize = 1; // choose the windogaussFiltWindSize of the gaussian filter

//Volume Rendering Parameters
int reductionFactor = 1;
cudaExtent volumeSize;
int cropOffset = 0;
float voxelThreshold;

//Bilateral Filter Paramters
int iterations;
float gaussian_delta;
float euclidean_delta;
float filter_radius;
int nthreads;

//Integer variables to keep track of Window IDs
int mainWindow;
int bScanWindow;
int fundusWindow;
int svfundusWindow;
int volumeWindow;
int linePlotWindow;
int svfundusColorWindow;
int svfundus2Window;

//Declaring the Textures to display for each window
GLuint mainTEX;
int mainTextureWidth = 1536;
int mainTextureHeight = 1024;
unsigned char *mainTexBuffer;

struct cudaGraphicsResource *bscanCudaRes;
GLuint bscanTEX;
GLuint bscanPBO;

struct cudaGraphicsResource *fundusCudaRes;
GLuint fundusTEX;
GLuint fundusPBO;

struct cudaGraphicsResource *svfundusCudaRes;
GLuint svfundusTEX;
GLuint svfundusPBO;

struct cudaGraphicsResource *volumeCudaRes;
GLuint volumeTEX;
GLuint volumePBO;

struct cudaGraphicsResource *svfundusColorCudaRes;
GLuint svfundusColorTEX;
GLuint svfundusColorPBO;

struct cudaGraphicsResource *svfundus2CudaRes;
GLuint svfundus2TEX;
GLuint svfundus2PBO;

struct cudaGraphicsResource *svfundus3CudaRes;
GLuint svfundus3TEX;
GLuint svfundus3PBO;


//Line Plot Attributes
GLint attribute_coord2d;
GLuint vbo;
struct point {
  GLfloat x;
  GLfloat y;
};
point *graph;
//--------------------//


//Declare Memory Pointers to be used for processing and display
unsigned short *buffer1;
float *h_procBuffer;
float *h_floatFrameBuffer;
float *h_floatVolumeBuffer;
float *d_FrameBuffer;
float *d_volumeBuffer;
float *d_svBuffer;
float *d_DisplayBuffer;
float *d_DisplayBuffer1;
float *d_DisplayBuffer2;
float *d_DisplayBuffer3;
float *d_DisplayBufferC;
float *d_DisplayBufferColor;
float *d_fundusBuffer;
float *d_fundusBuffer2;
float *d_fundusBuffer3;
float *readback;
float *d_phaseBuffer;

//For Callback which monitor the orientation of volume transformations
enum enMouseButton {mouseLeft, mouseMiddle, mouseRight, mouseNone} mouseButton = mouseNone;
int mouseX = 0, mouseY = 0;
float xAngle = 0.0f, yAngle = 0.0f;
float xTranslate = 0.0f, yTranslate = 0.0f, zTranslate = -4.0f;
float zoom = 1.0f;
float invViewMatrix[12];
float sum;
int clickctr = 0;
int bScanY[6];

dim3 blockSize(256);
dim3 gridSize;
//Save fundus
bool saveFile;
int saveFileCount = 0;
int saveNumFun = 200;
int saveFundusCount = 0;
//Round to nearest divisible value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Function to compute and display FPS
void computeFPS()
{
	const char* volumeMethod[] = {"Down Sizing", "Volume Cropping"};
	const char* samplingMethod[] = {"Linear", "Fast-Cubic", "Prefilter-Cubic"};
	const float updatesPerSec = 6.0f;
	static clock_t t1 = 0;
	static clock_t t2 = 0;
	static int counter = 0;
	static int countLimit = 0;

	if (counter++ > countLimit)
	{
		t2 = clock() - t1;
		t1 = clock();
		float framerate = 1000.0f * (float)counter / t2;
		char str[256];
		if (processRawData) {
			if(SensorlessAO){
				sprintf(str, "OCT Viewer, %s, %dx%dx%d: %3.1f fps, fundus intensity sum is :%3.5f", volumeMethod[displayMode], volumeWidth, volumeHeight, frames, framerate,sum);
			}else{
				sprintf(str, "OCT Viewer, %s, %s, %dx%dx%d: %3.1f fps", volumeMethod[displayMode], samplingMethod[sampMethod], volumeWidth, volumeHeight, frames, framerate);
			}
		} else {
			sprintf(str, "Display ONLY: %3.1f fps",framerate);
		}
		glutSetWindow(mainWindow);
		glutSetWindowTitle(str);
		countLimit = (int)(framerate / updatesPerSec);
		counter = 0;
	}
}

//For displaying processed Data ONLY
void copyFrameToFloat() 
{
	//Display Processed Data Only
	//First Calculate the coefficient in order to scale the data down to the float display range, 0-1
	//Processed data usually comes in 2-byte format, therefore the coefficient will be inverse of 2^16
	float coeff = 1/(pow(2.0f,16)-1);
	//memcpy(h_floatFrameBuffer, (float *)&buffer1[frameCount*(width*height)], width*height*framesPerBuffr*sizeof(unsigned short));
	for (int i = 0; i<width*height; i++) {
		h_floatFrameBuffer[i] = (float)buffer1[frameCount*(width*height) + i] * coeff;
	}
	frameCount = (frameCount + framesPerBuffr) % frames;
}

//For displaying processed volume ONLY
void copyVolumeToFloat() 
{
	//Display Processed Data Only
	//First Calculate the coefficient in order to scale the data down to the float display range, 0-1
	//Processed data usually comes in 2-byte format, therefore the coefficient will be inverse of 2^16
	float coeff = 1/(pow(2.0f,16)-1);
	//memcpy(h_floatFrameBuffer, (float *)&buffer1[frameCount*(width*height)], width*height*framesPerBuffr*sizeof(unsigned short));
	for (int i = 0; i<width*height*frames; i++) {
		h_floatVolumeBuffer[i] = pow((float)buffer1[i] * coeff,6);
	}
}


//Initialization for Main Window
//Main Texture is simply for background purposes
//Uncomment the file read lines for inserting customized background images
void initMainTexture()
{
	for (int i=0; i<mainTextureWidth*mainTextureHeight*3; i++)
		mainTexBuffer[i] = 0; //Light Gray Colour

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glGenTextures(1, &mainTEX);				//Generate the Open GL texture
	glBindTexture(GL_TEXTURE_2D, mainTEX); //Tell OpenGL which texture to edit
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mainTextureWidth, mainTextureHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, mainTexBuffer);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0); //Tell OpenGL which texture to edit
}



//Initialization for Bscan
void initBScanTexture()
{
		if (bscanPBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(bscanCudaRes);
			glDeleteBuffersARB(1, &bscanPBO);
			glDeleteTextures(1, &bscanTEX);
		}

		//Using ARB Method also works
		glGenBuffersARB(1, &bscanPBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanPBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanWidth * bscanHeight * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		cudaGraphicsGLRegisterBuffer(&bscanCudaRes, bscanPBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &bscanTEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, bscanTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, bscanWidth, bscanHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}


//Initialization for En Face
void initFundusTexture()
{
		if (fundusPBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(fundusCudaRes);
			glDeleteBuffersARB(1, &fundusPBO);
			glDeleteTextures(1, &fundusTEX);
		}
		//Using ARB Method also works
		glGenBuffersARB(1, &fundusPBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, fundusPBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumeHeight * frames * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &fundusCudaRes, fundusPBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &fundusTEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, fundusTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, volumeHeight, frames, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}


//Initialization for sv En Face
void initSvFundusTexture()
{
		if (svfundusPBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(svfundusCudaRes);
			glDeleteBuffersARB(1, &svfundusPBO);
			glDeleteTextures(1, &svfundusTEX);
		}
		//Using ARB Method also works
		glGenBuffersARB(1, &svfundusPBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundusPBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumeHeight * fundusFrames * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &svfundusCudaRes, svfundusPBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &svfundusTEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, svfundusTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, volumeHeight, fundusFrames, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}

void initSvFundusColorTexture()
{
		if (svfundusColorPBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(svfundusColorCudaRes);
			glDeleteBuffersARB(1, &svfundusColorPBO);
			glDeleteTextures(1, &svfundusColorTEX);
		}
		//Using ARB Method also works
		glGenBuffersARB(1, &svfundusColorPBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundusColorPBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumeHeight * fundusFrames * 3 * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &svfundusColorCudaRes, svfundusColorPBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &svfundusColorTEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, svfundusColorTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, volumeHeight, fundusFrames, 0, GL_RGB, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}
//Initialization for Frequency
void initSvFundus2Texture()
{
		if (svfundus2PBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(svfundus2CudaRes);
			glDeleteBuffersARB(1, &svfundus2PBO);
			glDeleteTextures(1, &svfundus2TEX);
		}
		//Using ARB Method also works
		glGenBuffersARB(1, &svfundus2PBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundus2PBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumeHeight * fundusFrames * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &svfundus2CudaRes, svfundus2PBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &svfundus2TEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, svfundus2TEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, volumeHeight, fundusFrames, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}
void initSvFundus3Texture()
{
		if (svfundus3PBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(svfundus3CudaRes);
			glDeleteBuffersARB(1, &svfundus3PBO);
			glDeleteTextures(1, &svfundus3TEX);
		}
		//Using ARB Method also works
		glGenBuffersARB(1, &svfundus3PBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundus3PBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumeHeight * fundusFrames * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &svfundus3CudaRes, svfundus3PBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &svfundus3TEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, svfundus3TEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, volumeHeight, fundusFrames, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}
//Initialization for Volume Window
void initVolumeTexture()
{
		if (volumePBO) {
			// unregister this buffer object from CUDA C
			cudaGraphicsUnregisterResource(volumeCudaRes);
			// delete old buffer
			glDeleteBuffersARB(1, &volumePBO);
			glDeleteTextures(1, &volumeTEX);
		}

		//Using ARB Method also works
		glGenBuffersARB(1, &volumePBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumePBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, subWindWidth * subWindHeight * 4 * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer( &volumeCudaRes, volumePBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &volumeTEX);		//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, volumeTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, subWindWidth, subWindHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
		gridSize = dim3(iDivUp(subWindWidth, blockSize.x), iDivUp(subWindHeight, blockSize.y));
}

//Initialization for Line Plot
void initlinePlotVBO(){
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, width*sizeof(point), graph, 
	GL_DYNAMIC_DRAW);
}


/*****************************************************************************************************************************/
/***********************************************  Open GL Callback Functions *************************************************/
/*****************************************************************************************************************************/

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
	{
    case 27:
        exit(0);
		break;
	case 'a':
		if (frameAveraging) {
			frameAveraging = false;
			printf("Frame Averaging OFF\n");
		} else {
			frameAveraging = true;
			printf("Frame Averaging ON\n");
		}
		break;
	case 'b':
		if (speckleVariance)
		{
			if (gaussFilt) {
				gaussFilt = false;
				printf("Gaussian Filter OFF\n");
			} else {
				gaussFilt = true;
				printf("Bilateral Filter ON\n");
			}
		}
		else
		{
			if (bilatFilt) {
				bilatFilt = false;
				printf("Bilateral Filter OFF\n");
			} else {
				bilatFilt = true;
				printf("Bilateral Filter ON\n");
			}
		}
		break;
	case 'c': // pixel-registration for each BM scan (3 adjacent frames)
		if (speckleVariance) {
			if (moCorr){
				moCorr = false;
				printf("Motion correction OFF\n");
			} else {
				moCorr = true;
				if (moCorr) {
					cudaDeviceSynchronize();
					fft2dPlanDestroy();
					fft2dPlanCreate(volumeHeight, volumeWidth,subPixelFactor);
					cudaDeviceSynchronize();
					getMeshgridFunc(volumeHeight, volumeWidth);
					cudaDeviceSynchronize();
					fftBufferDestroy();
					fftBufferCreate(volumeHeight, volumeWidth,subPixelFactor);
					printf("subPixelFactor:%d\n",subPixelFactor);
					cudaDeviceSynchronize();
				}
				printf("Motion correction ON\n");
			}
		}
		break;
	case 'd':
		acquireDC();
		break;
	case 'f':
		sampMethod = (sampMethod + 2) % 3;
		changeSamplingMethod(sampMethod);
		break;
	case 'v':
		if (notchFilt){
			notchFilt = false;
			printf("notchFilt OFF\n");
		} else {
			notchFilt = true;
			printf("notchFilt ON\n");
		}
		break;
	case'J':
		decreaseNotchGaussianSigmaH(volumeHeight, volumeWidth);
		break;
	case'K':
		increaseNotchGaussianSigmaH(volumeHeight, volumeWidth);
		break;
	case'I':
		decreaseNotchGaussianSigmaV(volumeHeight, volumeWidth);
		break;
	case'O':
		increaseNotchGaussianSigmaV(volumeHeight, volumeWidth);
		break;
	case '1':
		subPixelFactor = 1;
		printf("usfac__%d\n",subPixelFactor);
		break;
	case '2':
		subPixelFactor = 2;
		printf("usfac__%d\n",subPixelFactor);
		break;
	case '8':
		if (gaussFiltWindSize>21)
			printf("Gaussian size too big: is %d\n", gaussFiltWindSize);
		else
		{
			gaussFiltWindSize = gaussFiltWindSize + 2;
			printf("Gaussian size is: %d\n", gaussFiltWindSize);
		}
		//initGaussian(height, width,gaussFiltWindSize); 
		break;
	case '9':
		if (gaussFiltWindSize < 3)
			printf("Gaussian size too small: is %d\n", gaussFiltWindSize);
		else
		{
			gaussFiltWindSize = gaussFiltWindSize - 2;
			printf("Gaussian size is: %d\n", gaussFiltWindSize);
		}
		//initGaussian(height, width,gaussFiltWindSize); 
		break;
	case '0':
		MIP = !MIP;
		break;
	case 'q':
		sampMethod = (sampMethod + 1) % 3;
		changeSamplingMethod(sampMethod);
		break;
	case ',':
		decreaseDispVal();
		break;
	case '.':
		increaseDispVal();
		break;
	case 'r': //Reset all viewing windows to default
		sampMethod = 0;
		xAngle = 0.0f;
		yAngle = 0.0f;
		xTranslate = 0.0f;
		yTranslate = 0.0f;
		zTranslate = -4.0f;
		zoom = 1.0f;
		resetAllToInitialValues(sampMethod);
		break;
	case '-':
		decreaseMinVal();
		break;
	case '=':
		increaseMinVal();
		break;
	case '[':
		decreaseMaxVal();
		break;
	case ']':
		increaseMaxVal();
		break;
	case 'j':
		euclidean_delta -=0.1f;
		printf("euclidean_delta: %0.1f\n", euclidean_delta);
		break;
	case 'k':
		euclidean_delta += 0.1f;
		printf("euclidean_delta: %0.1f\n", euclidean_delta);
		break;
	case ';':
		if (voxelThreshold<=0.000f) {
			printf("Voxel Threshold has reached the minimum of 0.000!\n");
		} else {
			voxelThreshold-=0.002f;
			printf("Voxel Threshold is %0.3f\n", voxelThreshold);
		}
		break;
	case 'g':
		filter_radius -=1;
		printf("filter_radius: %d\n", filter_radius);
		break;
	case 'h':
		filter_radius +=1;
		printf("filter_radius: %d\n", filter_radius);
		break;
	case '\'':
		if (voxelThreshold>=0.200f) {
			printf("Voxel Threshold has reached the maximum of 0.200!\n");
		} else {
			voxelThreshold+=0.002f;
			printf("Voxel Threshold is %0.3f\n", voxelThreshold);
		}
		break;
	
	case 's':
		if (speckleVariance) {
			setMinMaxVal(1.8f,4.3f,2.0f);//The scaling factor 20 here was chosen for display purpose
			speckleVariance = false;
			printf("Speckle Variance OFF\n");

		} else {
			setMinMaxVal(2.55f,4.05f,12.0f);//The scaling factor 20 here was chosen for display purpose
			speckleVariance = true;
			printf("Speckle Variance ON\n");
		}
		cudaDeviceSynchronize();
		glutSetWindow(fundusWindow);
		initFundusTexture();
		cudaDeviceSynchronize();
		frameCount = 0;
		break;
	
	case 'n':
		decreaseFundusCoeff();
		break;
	case 'm':
		increaseFundusCoeff();
		break;
	case 't':
		if(svStartFrame >=2)
		{ printf("exceed maximum...%d",svStartFrame);
		}
		else{
			svStartFrame += 1;
			printf("current starting frame...%d",svStartFrame);
		}
		break;
	case 'y':
		if(svStartFrame <=0)
		{printf("reach minimum...%d",svStartFrame);}
		else{
			svStartFrame -= 1;
			printf("current starting frame...%d",svStartFrame);}
		break;
	case 'l':
		displayLine = !displayLine;
		break;
	case 'z':
		volumeProcess = !volumeProcess;
		frameCount = 0;
		break;
	case 'w':
		if (saveFile){}
				
			else 
				saveFile = true;
		saveFundusCount = 0;
		break;
    default:
        break;
    }
    glutPostRedisplay();
}


//Special Keyboard Functions
void specialKeyboard(int key, int x, int y)
{
	int offsetIncr = 16;
	int sizeIncr = 64; 
	int minSizeThres = 256;

	int maxReduction = 16;
	int minReduction = 2;

	bool volumeModified = false;
	bool bscanModified = false;
	bool bilatFiltModified = false;
	bool fundusModified = false;

	switch (key)
	{
/********************  END_KEY ***********************/
	case GLUT_KEY_END:
		if (displayMode == Crop) {
			if (volumeWidth!= width) {
				printf("Warning: Offset has been reset to zero to compensate for crop size increase!\n");
				cropOffset = 0;
				volumeWidth = width;
			} else {
				volumeWidth = minSizeThres;
			}

			volumeModified = true;
			bscanModified = true;
			bilatFiltModified = true;
		} 
		break;
/********************  HOME_KEY ***********************/
	case GLUT_KEY_HOME:
		if (volumeRender) {
			if (displayMode == Crop) {
				displayMode = DownSize;
				reductionFactor = minReduction;
				volumeWidth = width/reductionFactor;
				volumeHeight = height/reductionFactor;
				volumeModified = true;

			} else if (displayMode == DownSize) {
				displayMode = Crop;
				volumeWidth = minSizeThres;
				volumeHeight = height;
				volumeModified = true;
			}

			bscanModified = true;
			fundusModified = true;
			bilatFiltModified = true;
		}
		break;

/*********************  UP_KEY ************************/
	case GLUT_KEY_UP :
		if (displayMode == Crop) {
			if (cropOffset + volumeWidth + offsetIncr > width)
			{
				printf("Error: Unable to increase offset, Max Offset has been reached!!\n");
			} 
			else {
				cropOffset += offsetIncr;
			}
		}
		break;

/********************  DOWN_KEY ***********************/
	case GLUT_KEY_DOWN:
		if (displayMode == Crop) {
			if (cropOffset - offsetIncr < 0)
			{
				printf("Error: Unable to decrease offset, Zero Offset has been reached!!\n");
			} 
			else {
				cropOffset -= offsetIncr;
			}
		}
		break;

/*******************  RIGHT_KEY ***********************/
	case GLUT_KEY_RIGHT:
		if (displayMode == Crop) {
			if (volumeWidth + sizeIncr > width)
			{
				printf("Error: Maximum resolution has been reached!\n");
			} 
			else {
				if (cropOffset + volumeWidth + sizeIncr > width) {
					cropOffset = 0;
					printf("Warning: Offset has been reset to zero to compensate for crop size increase!\n");
				}
				volumeWidth += sizeIncr;
				volumeModified = true;
			}
		}
		else if (displayMode == DownSize) {
			if (reductionFactor==minReduction) {
				printf("Error: Minimum downsize Factor has been reached.\n For full resolution, press 'Home' to switch into Crop Mode.\n\n");
			} else {
				reductionFactor >>= 1;
				volumeWidth = width/reductionFactor;
				volumeHeight = height/reductionFactor;
				volumeModified = true;
			}
			fundusModified = true;
		}

		bscanModified = true;
		bilatFiltModified = true;

		break;

/*********************  LEFT_KEY ***********************/
	case GLUT_KEY_LEFT:
		if (displayMode == Crop) {
			if (volumeWidth - sizeIncr < minSizeThres)
			{
				printf("Error: Minimum allowed resolution has been reached!\n");
			} 
			else {
				volumeWidth -= sizeIncr;
				volumeModified = true;
			}
		}
		else if (displayMode == DownSize) {
			if (reductionFactor==maxReduction) {
				printf("Error: Maximum downsize Factor has been reached.\n\n");
			} else {
				reductionFactor <<= 1;
				volumeWidth = width/reductionFactor;
				volumeHeight = height/reductionFactor;
				volumeModified = true;
			}
			fundusModified = true;
		}

		bscanModified = true;
		bilatFiltModified = true;

		break;
	default:
		break;
	}


	//Actions for each Modification
	//B-scan Modification
	if (bscanModified) {
		glutSetWindow(bScanWindow);
		bscanWidth = volumeWidth;
		bscanHeight = volumeHeight;
		initBScanTexture();

		if (moCorr) {
			cudaDeviceSynchronize();
			fft2dPlanDestroy();
			fft2dPlanCreate(volumeHeight, volumeWidth,subPixelFactor);
			cudaDeviceSynchronize();
		}
	}

	//Bilateral Filter Modification
	if (bilatFilt && bilatFiltModified) {
		/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
		/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
		/**/freeFilterTextures(); //Texture will be reinitialized when initFilterTextures is recalled
		/**/updateGaussian(gaussian_delta, filter_radius);
		/*************************************************************************************************************************/
	}

	//En Face View Modification
	if (fundusRender && fundusModified) {
		glutSetWindow(fundusWindow);
		initFundusTexture();
	}

	//Volume Rendering Size Modification
	if (volumeRender && volumeModified) {
		/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
		/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
		/**/volumeSize = make_cudaExtent(volumeWidth, volumeHeight, frames);
		/**/freeVolumeBuffers();
		cudaMemset( d_volumeBuffer, 0, volumeWidth * volumeHeight * frames * sizeof(float));
		/**/initRayCastCuda(d_volumeBuffer, volumeSize, cudaMemcpyDeviceToDevice);
		/*************************************************************************************************************************/
	}else if ((slowBscan || volumeProcess) && volumeModified) {
		cudaMemset( d_volumeBuffer, 0, volumeWidth * volumeHeight * frames * sizeof(float));
	}

	glutPostRedisplay();
}//END OF SPECIAL KEY CALLBACKS


//RESIZE FUNCTION
//This defines what happens to the subwindows when resizing
void resize(int w, int h) {

	windowWidth = w;
	windowHeight = h;

	if (glutGetWindow() == mainWindow) {
		glViewport(0, 0, windowWidth, windowHeight);

		if (fundusRender) {
			subWindWidth = w/3;
		} else {
			subWindWidth = w;
		}

		subWindHeight = h/2;

		glutSetWindow(bScanWindow);
		glutPositionWindow(0, 0);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(linePlotWindow);
		glutPositionWindow(0, subWindHeight);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(fundusWindow);
		glutPositionWindow(subWindWidth, 0);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(svfundusWindow);
		glutPositionWindow(subWindWidth*2, 0);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(svfundusColorWindow);
		glutPositionWindow(subWindWidth*2, subWindHeight);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(svfundus2Window);
		glutPositionWindow(subWindWidth, subWindHeight);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(volumeWindow);
		glutPositionWindow(subWindWidth, subWindHeight);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);
		initVolumeTexture();
		gridSize = dim3(iDivUp(subWindWidth, blockSize.x), iDivUp(subWindHeight, blockSize.y));
	}

	if (glutGetWindow() == volumeWindow) {
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
		glutPostRedisplay();
	}
}


//Display Main is for Background Colour, the area where subwindows do not occupy
void displayMain() 
{
		glLoadIdentity();
		glRotatef(-90.0f, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindTexture(GL_TEXTURE_2D, mainTEX);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mainTextureWidth, mainTextureHeight, GL_BGR, GL_UNSIGNED_BYTE, mainTexBuffer);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glutSwapBuffers();
}



void displayBscan() 
{
	glLoadIdentity();

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanPBO);
	glBindTexture(GL_TEXTURE_2D, bscanTEX);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bscanWidth, bscanHeight, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glColor3f(1.0, 1.0, 1.0);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	if (Cscanselector) {
		float aScanCoord_0 = 0.0f;
		aScanCoord_0 = (float)(bScanY[0])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		float aScanCoord_1 = 0.0f;
		aScanCoord_1 = (float)(bScanY[1])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		glLineWidth(2.5);
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
			glVertex3f(-1.0, aScanCoord_0, 0.0);
			glVertex3f(1.0, aScanCoord_0, 0.0);
			glVertex3f(-1.0, aScanCoord_1, 0.0);
			glVertex3f(1.0, aScanCoord_1, 0.0);
			
		glEnd();
		float aScanCoord_2 = 0.0f;
		aScanCoord_2 = (float)(bScanY[2])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		float aScanCoord_3 = 0.0f;
		aScanCoord_3 = (float)(bScanY[3])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		glLineWidth(2.5);
		glColor3f(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
			glVertex3f(-1.0, aScanCoord_2, 0.0);
			glVertex3f(1.0, aScanCoord_2, 0.0);
			glVertex3f(-1.0, aScanCoord_3, 0.0);
			glVertex3f(1.0, aScanCoord_3, 0.0);	
		glEnd();

		float aScanCoord_4 = 0.0f;
		aScanCoord_4 = (float)(bScanY[4])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		float aScanCoord_5 = 0.0f;
		aScanCoord_5 = (float)(bScanY[5])*-2.0f / (float)(glutGet(GLUT_WINDOW_HEIGHT)) + 1.0f;
		glLineWidth(2.5);
		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
			glVertex3f(-1.0, aScanCoord_4, 0.0);
			glVertex3f(1.0, aScanCoord_4, 0.0);
			glVertex3f(-1.0, aScanCoord_5, 0.0);
			glVertex3f(1.0, aScanCoord_5, 0.0);	
		glEnd();
	}
	glutSwapBuffers();
}


void displayFundus() {
		glLoadIdentity();
		glRotatef(90, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, fundusPBO);
	glBindTexture(GL_TEXTURE_2D, fundusTEX);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volumeHeight, frames, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glColor3f(1.0, 1.0, 1.0);	
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

	//GL_PROJECTION COORDINATES
		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	if (fundusRender && displayLine) {
		float bScanCoord = 0.0f;
		bScanCoord = (float)(bScanFrameCount)*-2.0f / (float)(frames) + 1.0f;
		glLineWidth(2.5);
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
			glVertex3f(bScanCoord, -1.0, 0.0);
			glVertex3f(bScanCoord, 1.0, 0.0);
		glEnd();
	}

	glutSwapBuffers();
}

void displaySvFundus() {
		glLoadIdentity();
		glRotatef(90, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundusPBO);
	glBindTexture(GL_TEXTURE_2D, svfundusTEX);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volumeHeight, fundusFrames, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glColor3f(1.0, 1.0, 1.0);	
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

	//GL_PROJECTION COORDINATES
		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	//if (fundusRender && displayLine) {
	//	float bScanCoord = 0.0f;
	//	bScanCoord = (float)(bScanFrameCount)*-2.0f / (float)(frames) + 1.0f;
	//	glLineWidth(2.5);
	//	glColor3f(1.0, 0.0, 0.0);
	//	glBegin(GL_LINES);
	//		glVertex3f(bScanCoord, -1.0, 0.0);
	//		glVertex3f(bScanCoord, 1.0, 0.0);
	//	glEnd();
	//}

	glutSwapBuffers();
}

void displaySvFundusColor() {
		glLoadIdentity();
		glRotatef(90, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundusColorPBO);
	glBindTexture(GL_TEXTURE_2D, svfundusColorTEX);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volumeHeight, fundusFrames, GL_RGB, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

	//GL_PROJECTION COORDINATES
		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers();
}
void displaySvFundus2() {
		glLoadIdentity();
		glRotatef(90, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, svfundus2PBO);
	glBindTexture(GL_TEXTURE_2D, svfundus2TEX);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volumeHeight, fundusFrames, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

	//GL_PROJECTION COORDINATES
		glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1, 1); glVertex2f(-1.0,  1.0);
		glTexCoord2f(1, 0); glVertex2f( 1.0,  1.0);
		glTexCoord2f(0, 0); glVertex2f( 1.0, -1.0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers();
}
void displayVolume() {
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		glLoadIdentity();
		glRotatef(-xAngle, 1.0f, 0.0f, 0.0f);
		glRotatef(yAngle, 0.0f, 1.0f, 0.0f);
		glTranslatef(xTranslate, -yTranslate, -zTranslate);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	/************* This projection matrix configuration is from NVIDIA's volumeRender_kernel.cu at the following link: *******/
	/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
	invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];
	/*************************************************************************************************************************/

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, volumePBO);
	glBindTexture(GL_TEXTURE_2D, volumeTEX);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, subWindWidth, subWindHeight, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);

	//GL_MODELVIEW COORDINATES
		glTexCoord2f(0, 0); glVertex2f(1, 0);
		glTexCoord2f(1, 0); glVertex2f(1, 1);
		glTexCoord2f(1, 1); glVertex2f(0, 1);
		glTexCoord2f(0, 1); glVertex2f(0, 0);

    glEnd();
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers();
}


void displaylinePlot() {
	
	glClear(GL_COLOR_BUFFER_BIT);
	glBufferData(GL_ARRAY_BUFFER, width*sizeof(point), graph, 
	GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	 
	glEnableVertexAttribArray(attribute_coord2d);
	glVertexAttribPointer(
	  attribute_coord2d,   // attribute
	  2,                   // number of elements per vertex, here (x,y) 
	  GL_FLOAT,            // the type of each element
	  GL_FALSE,            // take our values as-is
	  0,                   // no space between values
	  0                    // use the vertex buffer object
	);
	 
	glDrawArrays(GL_LINE_STRIP, 0, width);
	glutSwapBuffers();
}

//Lineplot to be displayed
void updatelinePlot(){
	int lineScaleFactor = 2048; //This should be half the bit depth
	for(int i = 0; i < width; i++) {	  
		float x = (i-(float)width/2)/(float)(width/2);
		graph[i].x = x;
		if (volumeRender || fundusRender) {
			graph[i].y = (float)(buffer1[frameCount*width*height + width*height/2 + i])/lineScaleFactor - 1;
		} else {
			graph[i].y = (float)(buffer1[width*height/2 + i])/lineScaleFactor - 1;
		}
	}
}


__global__ void syncVolumeGL() {}

//Event to be executed on every iteration
void timerEvent(int value)
{
	// a-scan view --- display 4
	updatelinePlot();
	glutSetWindow(linePlotWindow);
	glutPostRedisplay();

	if (!volumeRender && !volumeProcess) {

		if (speckleVariance)
		{
			size_t size;
			int inputFrameCount = 0;
			int bScanFrameCount2 = 0;// for sv purpose
			int fundusPosition = 0;
			int svFramesBuffer = 0;
			if ((frameCount + framesPerBuffr +svStartFrame) <=(frames-framesPerBuffr)){
				frameCount = (frameCount + framesPerBuffr) % (frames);
			}
			else
			frameCount = 0;

			fundusPosition = ((frameCount + frames-framesPerBuffr) % frames)/3;
			svFramesBuffer = framesPerBuffr/3;
			if (autoBscan) {
				if (slowBscan) {
					bScanFrameCount = (bScanFrameCount + 1) % (frames-framesPerBuffr);
					bScanFrameCount2 = bScanFrameCount / 3 * 3;
					if ((frameCount-framesPerBuffr)< 0 ) 
					inputFrameCount = (frameCount -framesPerBuffr+frames)%frames;
					else
					inputFrameCount = (frameCount -framesPerBuffr)%frames;
				} else {
					bScanFrameCount = 0;
					inputFrameCount = 0;
				}
			} else {
				if (slowBscan) {
					bScanFrameCount2 = bScanFrameCount / 3 * 3;
					if ((frameCount-framesPerBuffr)< 0 ) 
					inputFrameCount = (frameCount -framesPerBuffr+frames)%frames;
					else
					inputFrameCount = (frameCount -framesPerBuffr)%frames;
				} else {
					inputFrameCount = 0;
				}
			}

			//Mapping resources for display
			cudaGraphicsMapResources(1,&bscanCudaRes,0);  // bscan   --- display 1 
			cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);
			cudaGraphicsMapResources(1,&fundusCudaRes,0); // original en-face --- display 2
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, fundusCudaRes);
			cudaGraphicsMapResources(1,&svfundusCudaRes,0); // svOCT en-face for selected reigion in red--- display 3
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer1, &size, svfundusCudaRes);
			cudaGraphicsMapResources(1,&svfundus2CudaRes,0); // svOCT en-face for selected reigion in green--- display 5
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer2, &size, svfundus2CudaRes);
			cudaGraphicsMapResources(1,&svfundus3CudaRes,0); // svOCT en-face for selected reigion in blue--- no display
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer3, &size, svfundus3CudaRes);			
			cudaGraphicsMapResources(1,&svfundusColorCudaRes,0); // combined color-coded svOCT en-face --- display 6
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBufferC, &size, svfundusColorCudaRes);
			
			//cudaPipeline for OCT processing
			if (displayMode == Crop) {
				cudaPipeline( &buffer1[(frameCount+svStartFrame)*(width*height)], d_volumeBuffer,d_phaseBuffer, inputFrameCount, 1,  cropOffset, volumeWidth,false);
			}	
			//Joined Kernels, One Kernel for PostFFT and Copy
			else if (displayMode == DownSize) {
				cudaPipeline( &buffer1[(frameCount+svStartFrame)*(width*height)], d_volumeBuffer,d_phaseBuffer, 0, reductionFactor, NULL, NULL,false);
			}
			// Local Batch Pixel-registration 
		    if (moCorr)
			{
				dftregistration(&d_volumeBuffer[(inputFrameCount)*(volumeWidth*height)], subPixelFactor,volumeWidth, bscanHeight, 3, 0);
			}
			
			// Fundus view --- display 2
			cudaRenderFundus(d_DisplayBuffer, d_volumeBuffer, volumeWidth, volumeHeight, framesPerBuffr, inputFrameCount, true,fundusOffset, fundusWidth,MIP);			
			cudaGraphicsUnmapResources(1,&fundusCudaRes,0);
			glutSetWindow(fundusWindow);
			glutPostRedisplay();

			// speckle variance calculation
			speckleVar(&d_volumeBuffer[(inputFrameCount)*(volumeWidth*height)], d_FrameBuffer, d_svBuffer,  volumeWidth, bscanHeight, 3, 0,frameAveraging);
			// svOCT bscan view --- display 1
			copySingleFrame(d_volumeBuffer, d_FrameBuffer,  volumeWidth, bscanHeight, bScanFrameCount2);
			cudaGraphicsUnmapResources(1,&bscanCudaRes,0);
			glutSetWindow(bScanWindow);
			glutPostRedisplay();

			if (fundusRender) {
				cudaSvRenderFundus(d_fundusBuffer,d_volumeBuffer, volumeWidth, volumeHeight,svFramesBuffer, fundusPosition,true,fundusOffset,fundusWidth,MIP);	
				cudaMemcpy(d_DisplayBuffer1, d_fundusBuffer, volumeHeight*fundusFrames*sizeof(float), cudaMemcpyDeviceToDevice);
				
				if (notchFilt)
					notchFilterGPU(d_DisplayBuffer1,volumeHeight,fundusFrames);
				if (gaussFilt)
					gaussianFilterGPU(d_DisplayBuffer1,volumeHeight,fundusFrames,gaussFiltWindSize);

				setColor<<<fundusFrames*volumeHeight/256,256>>>(d_DisplayBufferColor, d_DisplayBuffer1,0);
				// sv en-face view (red) --- display 3
				cudaGraphicsUnmapResources(1,&svfundusCudaRes,0);
				glutSetWindow(svfundusWindow);
				glutPostRedisplay();

				//2nd selected region
				cudaSvRenderFundus(d_fundusBuffer2,d_volumeBuffer, volumeWidth, volumeHeight,svFramesBuffer, fundusPosition,true,fundusOffset2,fundusWidth2,MIP);	
				cudaMemcpy(d_DisplayBuffer2, d_fundusBuffer2, volumeHeight*fundusFrames*sizeof(float), cudaMemcpyDeviceToDevice);
					
				if (notchFilt)
					notchFilterGPU(d_DisplayBuffer2,volumeHeight,fundusFrames);
				if (gaussFilt)
					gaussianFilterGPU(d_DisplayBuffer2,volumeHeight,fundusFrames,gaussFiltWindSize);		

				setColor<<<fundusFrames*volumeHeight/256,256>>>(d_DisplayBufferColor, d_DisplayBuffer2,1);
				
				// sv en-face view (green) --- display 5			
				cudaGraphicsUnmapResources(1,&svfundus2CudaRes,0);
				glutSetWindow(svfundus2Window);
				glutPostRedisplay();

				//3rd selected region	
				cudaSvRenderFundus(d_fundusBuffer3,d_volumeBuffer, volumeWidth, volumeHeight,svFramesBuffer, fundusPosition,true,fundusOffset3,fundusWidth3,MIP);	
				cudaMemcpy(d_DisplayBuffer3, d_fundusBuffer3, volumeHeight*fundusFrames*sizeof(float), cudaMemcpyDeviceToDevice);
				if (notchFilt)
					notchFilterGPU(d_DisplayBuffer3,volumeHeight,fundusFrames);
				if (gaussFilt) 
					gaussianFilterGPU(d_DisplayBuffer3,volumeHeight,fundusFrames,gaussFiltWindSize);
				setColor<<<fundusFrames*volumeHeight/256,256>>>(d_DisplayBufferColor, d_DisplayBuffer3,2);  

				// sv color en-face view (combined) --- display 6
				cudaMemcpy(d_DisplayBufferC,d_DisplayBufferColor,volumeHeight*fundusFrames*3*sizeof(float),cudaMemcpyDeviceToDevice);
				cudaGraphicsUnmapResources(1,&svfundus3CudaRes,0);
				cudaGraphicsUnmapResources(1,&svfundusColorCudaRes,0);
				glutSetWindow(svfundusColorWindow);
				glutPostRedisplay();
			}
		}
		
		else {
				if (processRawData && !fundusRender) {
					size_t size;
					cudaGraphicsMapResources(1,&bscanCudaRes,0);
					cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);
					cudaPipeline( buffer1, NULL, NULL, 0, 1, cropOffset, bscanWidth,false);
					if (frameAveraging) {
						frameAvg(NULL, d_FrameBuffer,  bscanWidth, bscanHeight, framesToAvg, 0);
					} else {
						copySingleFrame(NULL, d_FrameBuffer,  bscanWidth, bscanHeight, 0);
					}

					if (bilatFilt) {
						/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
						/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
						initTexture(width, height, d_FrameBuffer);
						bilateralFilter(d_FrameBuffer, bscanWidth, bscanHeight, euclidean_delta, filter_radius, iterations, nthreads,bilFilFactor);
						/*************************************************************************************************************************/
					}

					cudaGraphicsUnmapResources(1,&bscanCudaRes,0);
					glutSetWindow(bScanWindow);
					glutPostRedisplay();
				}
				else if (processRawData && fundusRender) {

						int inputFrameCount;
						int displayBscanCount;
						int prevframeCount = frameCount;
						size_t size;
						frameCount = (frameCount + framesPerBuffr) % frames;
	
						if (autoBscan) {
							if (slowBscan) {
								bScanFrameCount = (bScanFrameCount + 1) % (frames-framesPerBuffr);
								displayBscanCount = bScanFrameCount;
								inputFrameCount = prevframeCount;
							} else {
								bScanFrameCount = frameCount;
								displayBscanCount = 0;
								inputFrameCount = 0;
							}
						} else {
							if (slowBscan) {
								displayBscanCount = bScanFrameCount;
								inputFrameCount = prevframeCount;
							} else {
								displayBscanCount = bScanFrameCount%framesPerBuffr;
								inputFrameCount = 0;
							}
						}
//printf("...frameCount%d.bScanFrameCount %d.displayBscanCount%d.inputFrameCount.%d...frames.%d\n",frameCount,bScanFrameCount,displayBscanCount,inputFrameCount,  frames);
						cudaPipeline( &buffer1[frameCount*(width*height)], d_volumeBuffer, d_phaseBuffer, inputFrameCount, 1, cropOffset, bscanWidth, processPhase);

						if (autoBscan || (bScanFrameCount >= frameCount && bScanFrameCount < frameCount + framesPerBuffr)) {
							cudaGraphicsMapResources(1,&bscanCudaRes,0);
							cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);

							if (frameAveraging) {
								frameAvg(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, framesToAvg, displayBscanCount);
							} else {
								copySingleFrame(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, displayBscanCount);
							}

							if (bilatFilt) {
								/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
								/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
								initTexture(bscanWidth, bscanHeight, d_FrameBuffer);
								bilateralFilter(d_FrameBuffer, bscanWidth, bscanHeight, euclidean_delta, filter_radius, iterations, nthreads,bilFilFactor);
								/*************************************************************************************************************************/
							}

							cudaGraphicsUnmapResources(1,&bscanCudaRes,0);
							glutSetWindow(bScanWindow);
							glutPostRedisplay();
						}

						if (fundusRender) { 

							cudaGraphicsMapResources(1,&fundusCudaRes,0);
							cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, fundusCudaRes);
							cudaRenderFundus(d_DisplayBuffer, d_volumeBuffer, volumeWidth, volumeHeight, framesPerBuffr, frameCount, true,fundusOffset, fundusWidth,MIP);
							
							if(SensorlessAO){
								readbackPrefix(readback,d_DisplayBuffer, height, frames);
								sum=0;
								for(int i=0; i<height*frames;i++)
									sum = sum+readback[i];
							}
							cudaGraphicsUnmapResources(1,&fundusCudaRes,0);
							glutSetWindow(fundusWindow);
							glutPostRedisplay();
						}
					}
					else {
						copyFrameToFloat();
					}
		}

	} else if (volumeRender || volumeProcess) {
				int displayBscanCount;
				size_t size ;
				if (autoBscan) {
					bScanFrameCount = (bScanFrameCount + 1) % (frames-framesPerBuffr);
					displayBscanCount = bScanFrameCount;
				} else {
					displayBscanCount = bScanFrameCount;
				}

				if (processRawData) {
				//Volume Rendering for Processing Raw Data
					int newI;
					int i = 0;
					for (int j=0; j<frames+framesPerBuffr; j+=framesPerBuffr) {
						i = j%frames;
						newI = (i+framesPerBuffr)%frames;

						if (volumeRender) {
		//Unjoined Kernels, Post FFT + Copy Volume
							if (displayMode == Crop) {
								cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, d_phaseBuffer,i, 1, cropOffset, volumeWidth, processPhase);
							} 
		//Joined Kernels, One Kernel for PostFFT and Copy
							else if (displayMode == DownSize) {
								cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, d_phaseBuffer, i, reductionFactor, NULL, NULL, processPhase);
							}
						} else if (fundusRender && !volumeRender) {
								cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, d_phaseBuffer, i, 1, 0, volumeWidth, processPhase);
						}
					}
					cudaGraphicsMapResources(1,&bscanCudaRes,0);
					cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);

					if (frameAveraging) {
						frameAvg(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, framesToAvg, displayBscanCount);
					} else {
						copySingleFrame(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, displayBscanCount);
					}

					if (bilatFilt) {
						/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
						/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
						initTexture(bscanWidth, bscanHeight, d_FrameBuffer);						
						bilateralFilter(d_FrameBuffer, bscanWidth, bscanHeight, euclidean_delta, filter_radius, iterations, nthreads,bilFilFactor);
						/*************************************************************************************************************************/
					}
					cudaGraphicsUnmapResources(1,&bscanCudaRes,0);
					glutSetWindow(bScanWindow);
					glutPostRedisplay();

					frameCount = (frameCount + framesToAvg) % (frames);

					if (volumeRender) {
						/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
						/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
						volumeSize = make_cudaExtent(volumeWidth, volumeHeight, frames);
						initRayCastCuda(d_volumeBuffer, volumeSize, cudaMemcpyDeviceToDevice);
						/*************************************************************************************************************************/
					}
				}

				if (volumeRender) {
				/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
				/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
				/******/copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
					cudaGraphicsMapResources(1,&volumeCudaRes,0);
					cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, volumeCudaRes);
					cudaMemset(d_DisplayBuffer, 0, size);
				/******/rayCast_kernel(	gridSize, blockSize, d_DisplayBuffer, windowWidth, 
									windowHeight, 0.05f, 1.0f, 0.0f, 1.0f, voxelThreshold);
					cudaGraphicsUnmapResources(1,&volumeCudaRes,0);
					glutSetWindow(volumeWindow);
					glutPostRedisplay();
				/*************************************************************************************************************************/
				}

				if (fundusRender) {
					cudaGraphicsMapResources(1,&fundusCudaRes,0);
					cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, fundusCudaRes);
					cudaRenderFundus(d_DisplayBuffer, d_volumeBuffer, volumeWidth, volumeHeight, frames, 0, false, fundusOffset, fundusWidth,MIP);
					if(SensorlessAO){
						readbackPrefix(readback,d_DisplayBuffer, height, frames);
						sum=0;
						for(int i=0; i<height*frames;i++)
							sum = sum+readback[i];
					}
				cudaGraphicsUnmapResources(1,&fundusCudaRes,0);
				glutSetWindow(fundusWindow);
				glutPostRedisplay();
			}
	}
	
	computeFPS();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) mouseButton = (enMouseButton)button;
	else mouseButton = mouseNone;
	switch (mouseButton)
	{
		case mouseLeft:
			if (glutGetWindow() == fundusWindow) {
				autoBscan = false;
				int fundusY;
				if (y>=glutGet(GLUT_WINDOW_HEIGHT)) {
					fundusY = glutGet(GLUT_WINDOW_HEIGHT);
				} else if (y<0) {
					fundusY = 0;
				} else {
					fundusY = y;
				}
				float tempRatio = 0;
				tempRatio = (float)fundusY / (float)glutGet(GLUT_WINDOW_HEIGHT);
				bScanFrameCount = (int)(tempRatio*(frames-framesToAvg));
			}
			
			if (glutGetWindow() == bScanWindow && Cscanselector) {
				
				if(clickctr ==6){
					bScanY[0]=0;
					bScanY[1]=0;
					bScanY[2]=0;
					bScanY[3]=0;
					bScanY[4]=0;
					bScanY[5]=0;
					clickctr =0;
					fundusWidth = volumeWidth;
					fundusOffset = 0;
					fundusWidth2 = volumeWidth;
					fundusOffset2 = 0;
					fundusWidth3 = volumeWidth;
					fundusOffset3 = 0;
				}
				else{
					int winHeight;
					winHeight = glutGet(GLUT_WINDOW_HEIGHT);
					
					tempRatio2 = (float)volumeWidth /winHeight;
					bScanY[clickctr] = y;	
					if(bScanY[0]> bScanY[1]){
						fundusWidth = (int)((bScanY[0]-bScanY[1])*tempRatio2);
						fundusOffset = (int)((winHeight-bScanY[0])*tempRatio2);
					}
					else{
						fundusWidth = (int)((bScanY[1]-bScanY[0])*tempRatio2);
						fundusOffset = (int)((winHeight-bScanY[1])*tempRatio2);	
					}

					if(bScanY[2]> bScanY[3]){
						fundusWidth2 = (int)((bScanY[2]-bScanY[3])*tempRatio2);
						fundusOffset2 = (int)((winHeight-bScanY[2])*tempRatio2);
					}
					else{
						fundusWidth2 = (int)((bScanY[3]-bScanY[2])*tempRatio2);
						fundusOffset2 = (int)((winHeight-bScanY[3])*tempRatio2);
					}

					if(bScanY[4]> bScanY[5]){
						fundusWidth3 = (int)((bScanY[4]-bScanY[5])*tempRatio2);
						fundusOffset3 = (int)((winHeight-bScanY[4])*tempRatio2);
					}
					else{
						fundusWidth3 = (int)((bScanY[5]-bScanY[4])*tempRatio2);
						fundusOffset3 = (int)((winHeight-bScanY[5])*tempRatio2);
					}
					clickctr++;
				}
				
			}
			
			break;
		case mouseRight:
			if (glutGetWindow() == fundusWindow)
				autoBscan = true;
			if (glutGetWindow() == bScanWindow){ 
				Cscanselector = !Cscanselector;
				if(!Cscanselector){
					clickctr =0;
					fundusWidth = volumeWidth;
					fundusOffset = 0;
					fundusWidth2 = volumeWidth;
					fundusOffset2 = 0;
					fundusWidth3 = volumeWidth;
					fundusOffset3 = 0;
					bScanY[0]=0;
					bScanY[1]=0;
					bScanY[2]=0;
					bScanY[3]=0;
					bScanY[4]=0;
					bScanY[5]=0;
				}
				
			}
			
			break;
		default:
			break;

	}
	mouseX = x;
	mouseY = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	switch (mouseButton)
	{
		case mouseLeft:
			if (glutGetWindow() == fundusWindow) {
				int fundusY;
				if (y>=glutGet(GLUT_WINDOW_HEIGHT)) {
					fundusY = glutGet(GLUT_WINDOW_HEIGHT);
				} else if (y<0) {
					fundusY = 0;
				} else {
					fundusY = y;
				}
				float tempRatio = 0;
				tempRatio = (float)fundusY / (float)glutGet(GLUT_WINDOW_HEIGHT);
				bScanFrameCount = (int)(tempRatio*(frames-framesToAvg));
			}
			if (glutGetWindow() == volumeWindow) {
				xAngle += x - mouseX;
				yAngle += y - mouseY;
			}
			break;
		case mouseRight:
			if (glutGetWindow() == volumeWindow) {
				xTranslate += 0.002f * (y - mouseY);
				yTranslate -= 0.002f * (x - mouseX);
			}
			break;
		case mouseMiddle:
			if (glutGetWindow() == volumeWindow) {
				zTranslate += 0.01f * (y - mouseY); //For 3D zooming
				zoom += 0.005f * (y - mouseY);		//For 2D zooming
			}
			break;
		default:
			break;
	}
	mouseX = x;
	mouseY = y;
	glutPostRedisplay();
}

void mouseWheel(int wheel, int direction, int x, int y)
{
	int direction2 = direction;
	int direction3 = direction;
	if(glutGetWindow() == bScanWindow && Cscanselector){
		if(bScanY[1]-direction>=0&&bScanY[1]-direction<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[0]-direction<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[0]-direction>=0){
			if (tempRatio2<1){
				fundusOffset = fundusOffset+direction;
				direction *=ceil(1/tempRatio2);}
			else{
				fundusOffset = fundusOffset+(int)(direction*tempRatio2);}
			
			bScanY[0] = bScanY[0]-direction;
			bScanY[1] = bScanY[1]-direction;
		}
		if(bScanY[3]-direction2>=0&&bScanY[3]-direction2<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[2]-direction2<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[2]-direction2>=0){
			if (tempRatio2<1){
				fundusOffset2 = fundusOffset2+direction2;
				direction2 *=ceil(1/tempRatio2);}
			else{
				fundusOffset2 = fundusOffset2+(int)(direction2*tempRatio2);}
			
			bScanY[2] = bScanY[2]-direction2;
			bScanY[3] = bScanY[3]-direction2;
		}
		if(bScanY[5]-direction3>=0&&bScanY[5]-direction3<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[4]-direction3<glutGet(GLUT_WINDOW_HEIGHT)&&bScanY[4]-direction3>=0){
			if (tempRatio2<1){
				fundusOffset3 = fundusOffset3+direction3;
				direction3 *=ceil(1/tempRatio2);}
			else{
				fundusOffset3 = fundusOffset3+(int)(direction3*tempRatio2);}

			bScanY[4] = bScanY[4]-direction3;
			bScanY[5] = bScanY[5]-direction3;
		}
	}
}
/*****************************************************************************************************************************/
/***********************************************  End of Open GL Callback Functions ******************************************/
/*****************************************************************************************************************************/
void cleanUp ()
{
	glutLeaveMainLoop();//Clean up GL textures
	glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &bscanTEX);
	glDeleteBuffersARB(1, &bscanPBO);
	cudaGraphicsUnmapResources(1,&bscanCudaRes,NULL);
	cudaGraphicsUnregisterResource(bscanCudaRes);

	glDeleteTextures(1, &fundusTEX);
	glDeleteBuffersARB(1, &fundusPBO);
	cudaGraphicsUnmapResources(1,&fundusCudaRes,NULL);
	cudaGraphicsUnregisterResource(fundusCudaRes);

	glDeleteTextures(1, &svfundusTEX);
	glDeleteBuffersARB(1, &svfundusPBO);
	cudaGraphicsUnmapResources(1,&svfundusCudaRes,NULL);
	cudaGraphicsUnregisterResource(svfundusCudaRes);
	
	glDeleteTextures(1, &volumeTEX);
	glDeleteBuffersARB(1, &volumePBO);
	cudaGraphicsUnmapResources(1,&volumeCudaRes,NULL);
	cudaGraphicsUnregisterResource(volumeCudaRes);

	//Free up buffers
	cudaHostUnregister(buffer1);
	free(buffer1);
	free(h_floatFrameBuffer);

	free(h_floatVolumeBuffer);

	//Free up Device Buffers
	cudaFree(d_volumeBuffer);
	cudaFree(d_FrameBuffer);
	cudaFree(d_DisplayBuffer);
	cudaFree(d_DisplayBuffer1);
	cudaFree(d_DisplayBuffer2);
	cudaFree(d_DisplayBuffer3);
	cudaFree(d_DisplayBufferColor);

	if (processRawData) {
		cleanUpCUDABuffers();
		freeFilterTextures();
	}
	if (volumeRender) {
		freeVolumeBuffers();
	}

	cudaDeviceReset();
}

/*************************************************************************************************************************
**************************************************************************************************************************
*************************************** External C Functions *************************************************************
**************************************************************************************************************************
*************************************************************************************************************************/
void initGLEvent(int argc, char** argv)
{
	//GL INITIALIZATION:
	glutInit(&argc, argv); //glutInit will initialize the GLUT library to operate with the Command Line
	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(windowWidth, windowHeight);
	mainWindow = glutCreateWindow("OCTViewer");
	glutDisplayFunc(displayMain);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutReshapeFunc(resize);
	glewInit();
	initMainTexture();

	//Initialize all the GL callback functions
	bScanWindow = glutCreateSubWindow(mainWindow, 0,0,subWindWidth,subWindHeight);
	glutDisplayFunc(displayBscan);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initBScanTexture();

	//Initialize Line Plot
	linePlotWindow = glutCreateSubWindow(mainWindow, 0,512,subWindWidth,subWindHeight);
	glutDisplayFunc(displaylinePlot);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glewInit();
	initlinePlotVBO();


	fundusWindow = glutCreateSubWindow(mainWindow, 512,0,subWindWidth,subWindHeight);
	glutDisplayFunc(displayFundus);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initFundusTexture();

	svfundusWindow = glutCreateSubWindow(mainWindow, 1024,0,subWindWidth,subWindHeight);
	glutDisplayFunc(displaySvFundus);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initSvFundusTexture();
	
	svfundus2Window = glutCreateSubWindow(mainWindow, 512,512,subWindWidth,subWindHeight);
	glutDisplayFunc(displaySvFundus2);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initSvFundus2Texture();
	initSvFundus3Texture();

	svfundusColorWindow = glutCreateSubWindow(mainWindow, 1024,512,subWindWidth,subWindHeight);
	glutDisplayFunc(displaySvFundusColor);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initSvFundusColorTexture();
	
	volumeWindow = glutCreateSubWindow(mainWindow, 512,512,subWindWidth,subWindHeight);
	glutDisplayFunc(displayVolume);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glutReshapeFunc(resize);
	glewInit();
	initVolumeTexture();
	
	initGaussian(height, width,gaussFiltWindSize);
	initNotchFiltVarAndPtrs(height,frames/3);

	//glutTimerFunc is a global callback function
	//Meaning it is not associated with any window
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
	//End of GL callback functions
//END OF GL INITIALIZATION


	if (volumeRender || volumeProcess || (fundusRender && slowBscan)) {

		cudaMalloc((void**)&d_fundusBuffer, height*frames*sizeof(float));
		cudaMemset(d_fundusBuffer, 0, height*frames*sizeof(float));
		cudaMalloc((void**)&d_fundusBuffer2, height*frames*sizeof(float));
		cudaMemset(d_fundusBuffer2, 0, height*frames*sizeof(float));
		cudaMalloc((void**)&d_fundusBuffer3, height*frames*sizeof(float));
		cudaMemset(d_fundusBuffer3, 0, height*frames*sizeof(float));
		cudaMalloc((void**)&d_DisplayBufferColor, height*frames*3*sizeof(float));
		cudaMemset(d_DisplayBufferColor,0,height*frames*3*sizeof(float));

		if (processRawData) { 
			

			cudaMalloc((void**)&d_volumeBuffer, width * height * frames * sizeof(float));

			cudaMemset( d_volumeBuffer, 0, width * height * frames * sizeof(float));
			cudaMalloc((void**)&d_svBuffer, width*height*sizeof(float));
			cudaMemset( d_svBuffer, 0, width*height*sizeof(float));
			if (processPhase)
			{
			cudaMalloc((void**)&d_phaseBuffer, width * height * frames * sizeof(float));
			cudaMemset( d_phaseBuffer, 0, width * height * frames * sizeof(float));
			}

		} else if (!processRawData) {
			/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
			/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
			/**/copyVolumeToFloat();
			/**/cudaExtent volumeSize = make_cudaExtent(width, height, frames);
			/**/initRayCastCuda((void *)h_floatVolumeBuffer, volumeSize, cudaMemcpyHostToDevice);
			/*************************************************************************************************************************/
		}
	} else {
		d_volumeBuffer = NULL;
	}

	/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
	/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
	/**/updateGaussian(gaussian_delta, filter_radius);
	/*************************************************************************************************************************/
	//glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
 //             GLUT_ACTION_GLUTMAINLOOP_RETURNS);
 //
	//atexit(cleanUp);
}


void runGLEvent() 
{
	glutMainLoopEvent();
	
}


void setBufferPtr ( unsigned short *h_buffer)
{
	buffer1 = h_buffer;
}

void setProcBufferPtr ( float *p_procbuffer) //"p_ for parameter
{
	h_procBuffer = p_procbuffer;
	cudaHostRegister(h_procBuffer, width * height * frames * sizeof(float), cudaHostRegisterDefault);
}

void setFrameSize(int frameSize)
{
	if (frames != frameSize) {
		frames = frameSize;
		if (fundusRender) {
			glutSetWindow(fundusWindow);
			initFundusTexture();
		}
		if (volumeRender) {
			/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
			/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
			/**/volumeSize = make_cudaExtent(volumeWidth, volumeHeight, frames);
			/**/freeVolumeBuffers();
			cudaMemset( d_volumeBuffer, 0, volumeWidth * volumeHeight * frames * sizeof(float));
			/**/initRayCastCuda(d_volumeBuffer, volumeSize, cudaMemcpyDeviceToDevice);
			/*************************************************************************************************************************/
		}
	}
}

void registerCudaHost()
{
	cudaHostRegister(buffer1, width * height * frames * sizeof(unsigned short), cudaHostRegisterDefault);
}

void fundusSwitch()
{
	if (fundusRender) {
		size_t size;
		cudaGraphicsMapResources(1,&fundusCudaRes,0);
		cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, fundusCudaRes);
		cudaMemset( d_DisplayBuffer, 0, size);
		cudaGraphicsUnmapResources(1,&fundusCudaRes,0);
		glutSetWindow(fundusWindow);
		glutPostRedisplay();
	}
	fundusRender = !fundusRender;
}

void volumeSwitch()
{
	if (volumeRender) {
		size_t size;
		/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
		/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
		/**/copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
		/*************************************************************************************************************************/
		cudaGraphicsMapResources(1,&volumeCudaRes,0);
		cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, volumeCudaRes);
		cudaMemset(d_DisplayBuffer, 0, size);
		cudaGraphicsUnmapResources(1,&volumeCudaRes,0);
		glutSetWindow(volumeWindow);
		glutPostRedisplay();
	}
	volumeRender = !volumeRender;
}

void memcpyFunctionProc()
{
	cudaMemcpy(h_procBuffer,d_volumeBuffer,width*height*frames*sizeof(float),cudaMemcpyDeviceToHost);
		//void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
}
void memcpyFunctionPhase()
{
	cudaMemcpy(h_procBuffer,d_phaseBuffer,width*height*frames*sizeof(float),cudaMemcpyDeviceToHost);
}
void setPhaseProcessTrue()
{
	processPhase = true;
}
void setPhaseProcessFalse()
{
	processPhase = false;
}


void initGLVarAndPtrs(	bool processData,
						bool volumeRend,
						bool fundRend,
						bool slowBsc,
						bool volumeProc,
						int frameWid, 
						int frameHei, 
						int framesPerBuff,
						int volFrames,
						int winWid,
						int winHei,
						int interpMethod,
						int volumeMode)
{
	cudaGLSetGLDevice(0);

	processRawData = processData;
	volumeRender = volumeRend;
	fundusRender = fundRend;
	frameAveraging = false;
	bilatFilt = false;
	autoBscan = true;
	Cscanselector=false;
	displayLine = true;
	slowBscan = slowBsc;
	volumeProcess = volumeProc;
	processPhase = false;
	SensorlessAO = true;
	width = frameWid;
	height = frameHei;
	frames = volFrames;
	fundusFrames = frames/3; //For svOCT, 3 is the BM scan size.
	if (frames%framesPerBuff != 0)
		frames -= framesPerBuff;
	
	subPixelFactor = 1;
	gaussFiltWindSize = 3;

	if (processRawData) {
		framesPerBuffr = framesPerBuff;
	} else {
		framesPerBuffr = 1;
	}

	if (!volumeRend && !fundRend && framesPerBuff<2) {
		framesToAvg = 1;
	} else {
		framesToAvg = 2;
	}

	windowWidth = winWid;
	windowHeight = winHei;
	subWindWidth = winWid/2;
	subWindHeight = winHei/2;
	voxelThreshold = 0.05f;

	sampMethod = interpMethod;

	if (volumeRender) {
		displayMode = (volumeDisplay)volumeMode;
		cropOffset = 0;

		if (processRawData) {
			d_FrameBuffer = 0;
			if (displayMode == Crop) {
				volumeWidth = width;
				volumeHeight = height;
				reductionFactor = 1;
			} else if (displayMode == DownSize) {
				reductionFactor = 2;
				volumeWidth = width/reductionFactor;
				volumeHeight = height/reductionFactor;
			}
		} else if (!processRawData) {
			h_floatVolumeBuffer = (float *)malloc(width * height * frames * sizeof(float));
			memset(h_floatVolumeBuffer, 0, width * height * frames * sizeof(float));
		}

		bscanWidth = volumeWidth;
		bscanHeight = volumeHeight;

	} else {
		if (!processRawData) {
			h_floatFrameBuffer = (float *)malloc(width * height * framesPerBuffr * sizeof(float));
			memset(h_floatFrameBuffer, 0, width * height * framesPerBuffr * sizeof(float));
		}
		bscanWidth = width;
		bscanHeight = height;
		volumeWidth = width;
		volumeHeight = height;
		fundusWidth = width;
		fundusOffset = 0;
		bScanY[0]=0;
		bScanY[1]=0;
		bScanY[2]=0;
		bScanY[3]=0;
		bScanY[4]=0;
		bScanY[5]=0;
	}

	mainTexBuffer = (unsigned char *) malloc (mainTextureWidth*mainTextureHeight*3);
	graph = (point *) malloc (width*sizeof(point));
	if(SensorlessAO)
		readback = (float *)malloc(height*frames*sizeof(float));
}