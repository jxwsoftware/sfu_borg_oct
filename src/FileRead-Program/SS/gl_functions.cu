/**********************************************************************************
Filename	: gl_functions.cu
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
//Include the windows.h for for using WINDOWS API Functions
#include <windows.h> //Include the windows.h for creating a window
#include <helper_cuda.h> //This is to perform CUDA safecall functions
#include <GL/glew.h> //Required to Generate GL Buffers
#include <GL/freeglut.h>
#include <cuda_gl_interop.h> 


#include "cuda_ProcHeader.cuh"


//Delay upon which timerevent is called
#define	REFRESH_DELAY	0 //ms


enum volumeDisplay {DownSize, Crop};
volumeDisplay displayMode = Crop;


//All initialization values are MEANINGLESS
//Do not worry about these initialized values!

//Boolean Variables to determine display and processing functionality
bool processData = true;
bool volumeRender = true;
bool fundusRender = true;
bool frameAveraging = true;
bool bilatFilt = false;

//Integer variables to determine display window sizes and resolution
int windowWidth = 512;
int windowHeight = 512;
int subWindWidth = 512;
int subWindHeight = 512;
int	width;
int height;
int frames;
int bscanWidth;
int bscanHeight;
int volumeWidth;
int volumeHeight;

//Processing and Bscan Display Parameters
int framesPerBuffr;
int framesToAvg = 3;
int frameCount = 0;

//Volume Rendering Parameters
int reductionFactor = 1;
cudaExtent volumeSize;
int cropOffset = 0;
float voxelThreshold;

//Bilateral Filter Paramters
int iterations = 1;
float gaussian_delta = 4;
float euclidean_delta = 0.1f;
int filter_radius = 5;
int nthreads = 256;

//Integer variables to keep track of Window IDs
int mainWindow;
int bScanWindow;
int fundusWindow;
int volumeWindow;

//Declaring the Textures to display for each window
GLuint mainTEX;
int mainTextureWidth = 1024;
int mainTextureHeight = 1024;
unsigned char *mainTexBuffer;

struct cudaGraphicsResource *bscanCudaRes;
GLuint bscanTEX;
GLuint bscanPBO;

struct cudaGraphicsResource *fundusCudaRes;
GLuint fundusTEX;
GLuint fundusPBO;

struct cudaGraphicsResource *volumeCudaRes;
GLuint volumeTEX;
GLuint volumePBO;

//Declare Memory Pointers to be used for processing and display
unsigned short *buffer1;
float *h_floatFrameBuffer;
float *h_floatVolumeBuffer;
float *d_FrameBuffer;
float *d_volumeBuffer;
float *d_DisplayBuffer;
unsigned int hTimer = 0;

//For Callback which monitor the orientation of volume transformations
enum enMouseButton {mouseLeft, mouseMiddle, mouseRight, mouseNone} mouseButton = mouseNone;
int mouseX = 0, mouseY = 0;
float xAngle = 0.0f, yAngle = 0.0f;
float xTranslate = 0.0f, yTranslate = 0.0f, zTranslate = -4.0f;
float zoom = 1.0f;
float invViewMatrix[12];

dim3 blockSize(256);
dim3 gridSize;


/************* This Functions have been modified from NVIDIA's volumeRender.cpp at the following link: *******************/
/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
/*************************************************************************************************************************/


void computeFPS()
{
	const char* volumeMethod[] = {"Down Sizing", "Volume Cropping"};

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
		if (processData) {




			sprintf(str, "OCT Viewer, %s, %dx%dx%d: %3.1f fps", volumeMethod[displayMode], volumeWidth, volumeHeight, frames, framerate);

		} else {
			sprintf(str, "Display ONLY: %3.1f fps",framerate);
		}

		glutSetWindow(mainWindow);
		glutSetWindowTitle(str);
		countLimit = (int)(framerate / updatesPerSec);
		counter = 0;
	}
}



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


// Initialization
//Main Texture is simply for background purposes
//Uncomment the file read lines for inserting customized background images
void initMainTexture()
{
	for (int i=0; i<mainTextureWidth*mainTextureHeight*3; i++)
		mainTexBuffer[i] = 150; //Light Gray Colour

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glGenTextures(1, &mainTEX);				//Generate the Open GL texture
	glBindTexture(GL_TEXTURE_2D, mainTEX); //Tell OpenGL which texture to edit
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mainTextureWidth, mainTextureHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, mainTexBuffer);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0); //Tell OpenGL which texture to edit
}


// Initialization
void initBScanTexture()
{
		if (bscanPBO) {
			checkCudaErrors(cudaGraphicsUnregisterResource(bscanCudaRes));
			glDeleteBuffersARB(1, &bscanPBO);
			glDeleteTextures(1, &bscanTEX);
		}

		//Using ARB Method also works
		glGenBuffersARB(1, &bscanPBO);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanPBO);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanWidth * bscanHeight * sizeof(float), 0, GL_STREAM_DRAW_ARB);
		//glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		cudaGraphicsGLRegisterBuffer(&bscanCudaRes, bscanPBO, cudaGraphicsMapFlagsNone);

		glGenTextures(1, &bscanTEX);				//Generate the Open GL texture
		glBindTexture(GL_TEXTURE_2D, bscanTEX); //Tell OpenGL which texture to edit
		glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, bscanWidth, bscanHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}


// Initialization
void initFundusTexture()
{
		if (fundusPBO) {
			checkCudaErrors(cudaGraphicsUnregisterResource(fundusCudaRes));
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
		glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, volumeHeight, frames, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
}

// Initialization
void initVolumeTexture()
{
		if (volumePBO) {
			checkCudaErrors(cudaGraphicsUnregisterResource(volumeCudaRes));
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
		glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, subWindWidth, subWindHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		//GL_LINEAR Allows the GL Display to perform Linear Interpolation AFTER processing
		//This means that when zooming into the image, the zoomed display will be much smoother
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
		gridSize = dim3(iDivUp(subWindWidth, blockSize.x), iDivUp(subWindHeight, blockSize.y));
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
	case 'd':
		acquireDC();
		break;
	case 'r': //Viewing Window to Default Orientation
		xAngle = 0.0f;
		yAngle = 0.0f;
		xTranslate = 0.0f;
		yTranslate = 0.0f;
		zTranslate = -4.0f;
		zoom = 1.0f;
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
	case ';':
		if (voxelThreshold<=0.000f) {
			printf("Voxel Threshold has reached the minimum of 0.000!\n");
		} else {
			voxelThreshold-=0.002f;
			printf("Voxel Threshold is %0.3f\n", voxelThreshold);
		}
		break;
	case '\'':
		if (voxelThreshold>=0.200f) {
			printf("Voxel Threshold has reached the maximum of 0.200!\n");
		} else {
			voxelThreshold+=0.002f;
			printf("Voxel Threshold is %0.3f\n", voxelThreshold);
		}
		break;
	case 'a':
		if (frameAveraging) {
			frameAveraging = false;
			printf("Frame Averaging OFF\n", voxelThreshold);
		} else {
			frameAveraging = true;
			printf("Frame Averaging ON\n", voxelThreshold);
		}
		break;
	case 'b':
		if (bilatFilt) {
			bilatFilt = false;
			printf("Bilateral Filter OFF\n", voxelThreshold);
		} else {
			bilatFilt = true;
			printf("Bilateral Filter ON\n", voxelThreshold);
		}
		break;
	case '`': //Manual Fundus switch on or off
		fundusSwitch();
		volumeSwitch();
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
	int sizeIncr = 16;
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
	}

	glutPostRedisplay();
}//END OF SPECIAL KEY CALLBACKS

//Resizing Window Callback Function
void resize(int w, int h) {

	windowWidth = w;
	windowHeight = h;

	if (glutGetWindow() == mainWindow) {
		glViewport(0, 0, windowWidth, windowHeight);
		subWindWidth = w/2;
		subWindHeight = h/2;

		glutSetWindow(bScanWindow);
		glutPositionWindow(0, 0);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(fundusWindow);
		glutPositionWindow(subWindWidth, 0);
		glutReshapeWindow(subWindWidth, subWindHeight);
		glViewport(0, 0, subWindWidth, subWindHeight);

		glutSetWindow(volumeWindow);
		glutPositionWindow(0, subWindHeight);
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
		glTranslatef(0.0f, -xTranslate, 0.0f);
		glScalef(1.0f, zoom, 0.1f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, bscanPBO);
	glBindTexture(GL_TEXTURE_2D, bscanTEX);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bscanWidth, bscanHeight, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
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


void displayFundus() {

		glLoadIdentity();
		glRotatef(90, 0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, fundusPBO);
	glBindTexture(GL_TEXTURE_2D, fundusTEX);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, volumeHeight, frames, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
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



void timerEvent(int value)
{
	if (!volumeRender && !fundusRender) {
		if (processData) {
			size_t size;
			checkCudaErrors(cudaGraphicsMapResources(1,&bscanCudaRes,0));
			cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);
			cudaPipeline( buffer1, NULL, 0, 1, 0, bscanWidth);
			if (frameAveraging) {
				frameAvg(NULL, d_FrameBuffer,  bscanWidth, bscanHeight, framesToAvg, 0);
			} else {
				copySingleFrame(NULL, d_FrameBuffer,  bscanWidth, bscanHeight, 0);
			}

			if (bilatFilt) {
				/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
				/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
				initTexture(width, height, d_FrameBuffer);
				bilateralFilter(d_FrameBuffer, width, height, euclidean_delta, filter_radius, iterations, nthreads);
				/*************************************************************************************************************************/
			}

			cudaGraphicsUnmapResources(1,&bscanCudaRes,0);
			glutSetWindow(bScanWindow);
			glutPostRedisplay();
		}
		else {
			copyFrameToFloat();
		}

	} else if (volumeRender || fundusRender) {
		size_t size ;
		if (processData) {
		//Volume Rendering for Processing Raw Data
			int newI;

			for (int i=0; i<frames; i+=framesPerBuffr) {
				newI = (i+framesPerBuffr)%frames;

				if (volumeRender) {
					//Cropping mode, which will display the volume in FULL resolution
					if (displayMode == Crop) {
						cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, i, 1, cropOffset, volumeWidth);
					} 
					//Downsize mode, which will display the volume in reduced resolution
					else if (displayMode == DownSize) {
						cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, i, reductionFactor, NULL, NULL);
					}
				} else if (fundusRender && !volumeRender) {
						cudaPipeline( &buffer1[newI*(width*height)], d_volumeBuffer, i, 1, 0, volumeWidth);
				}
			}

			checkCudaErrors(cudaGraphicsMapResources(1,&bscanCudaRes,0));
			cudaGraphicsResourceGetMappedPointer((void**) &d_FrameBuffer, &size, bscanCudaRes);
			if (frameAveraging) {
				frameAvg(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, framesToAvg, frameCount);
			} else {
				copySingleFrame(d_volumeBuffer, d_FrameBuffer,  bscanWidth, bscanHeight, frameCount);
			}

			if (bilatFilt) {
				/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
				/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
				initTexture(bscanWidth, bscanHeight, d_FrameBuffer);
				bilateralFilter(d_FrameBuffer, bscanWidth, bscanHeight, euclidean_delta, filter_radius, iterations, nthreads);
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
			checkCudaErrors(cudaGraphicsMapResources(1,&volumeCudaRes,0));
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, volumeCudaRes);
			checkCudaErrors(cudaMemset(d_DisplayBuffer, 0, size));
		/******/rayCast_kernel(	gridSize, blockSize, d_DisplayBuffer, windowWidth, 
							windowHeight, 0.05f, 1.0f, 0.0f, 1.0f, voxelThreshold);
			cudaGraphicsUnmapResources(1,&volumeCudaRes,0);
			glutSetWindow(volumeWindow);
			glutPostRedisplay();
		/*************************************************************************************************************************/
		}

		if (fundusRender) {
			checkCudaErrors(cudaGraphicsMapResources(1,&fundusCudaRes,0));
			cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, fundusCudaRes);
			cudaRenderFundus(d_DisplayBuffer, d_volumeBuffer, volumeWidth, volumeHeight, frames);
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

	mouseX = x;
	mouseY = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	switch (mouseButton)
	{
		case mouseLeft:
			if (glutGetWindow() == volumeWindow) {
				xAngle += x - mouseX;
				yAngle += y - mouseY;
			}
			break;
		case mouseRight:
			xTranslate += 0.002f * (y - mouseY);
			yTranslate -= 0.002f * (x - mouseX);
			break;
		case mouseMiddle:
			zTranslate += 0.01f * (y - mouseY); //For 3D zooming
			zoom += 0.005f * (y - mouseY);		//For 2D zooming
			break;
		case mouseNone:
		default:
			break;
	}
	mouseX = x;
	mouseY = y;
	glutPostRedisplay();
}

void mouseWheel(int wheel, int direction, int x, int y)
{
	//Currently MouseWheel does not have any functionality
	//This can be customized for any custom operations
}
/*****************************************************************************************************************************/
/***********************************************  End of Open GL Callback Functions ******************************************/
/*****************************************************************************************************************************/

void printUserManual() 
{
	printf("Ensure CAPS LOCK is OFF when using these hotkeys!\n");
	printf("'d'	- DC Subtraction\n");
	printf("'a'	- frame averaging\n");
	printf("'b'	- bilateral filtering\n");
	printf("'-'	- adjust constrast: decrease min value\n");
	printf("'='	- adjust constrast: increase min value\n");
	printf("'['	- adjust constrast: decrease max value\n");
	printf("']'	- adjust constrast: increase max value\n");
	printf("';'	- decrease volume rendering voxel threshold\n");
	printf("'''	- increase volume rendering voxel threshold\n");
	printf("'r'	- reset volume orientation\n");
	printf("'left'	- (Crop)Decrease volume size.  (Downsize)Increase downsizing\n");
	printf("'right'	- (Crop)Increase volume size.  (Downsize)Decrease downsizing\n");
	printf("'up'	- (Crop)Shift volume upward by increasing offset\n");
	printf("'down'	- (Crop)Shift volume downward by decreasing offset\n");
	printf("'home'	- (Volume Only) Switch between Crop and Downsize mode\n");
	printf("'end'	- (Crop)Switch from min size to max size\n");
	printf("'esc'	- Exit program\n");
	printf("End of Keyboard Functions.\n\n");
}

void cleanUp()
{
	//Clean up GL textures
	glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &bscanTEX);
	glDeleteBuffersARB(1, &bscanPBO);
	cudaGraphicsUnmapResources(1,&bscanCudaRes,NULL);
	cudaGraphicsUnregisterResource(bscanCudaRes);

	glDeleteTextures(1, &fundusTEX);
	glDeleteBuffersARB(1, &fundusPBO);
	cudaGraphicsUnmapResources(1,&fundusCudaRes,NULL);
	cudaGraphicsUnregisterResource(fundusCudaRes);
	
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
	
	if (processData) {
		cleanUpCUDABuffers();
		freeFilterTextures();
	}
	if (volumeRender) {
		freeVolumeBuffers();
	}
	
	cudaDeviceReset();
}

/*************************************************************************************************************************
*************************************** External C Functions *************************************************************
*************************************************************************************************************************/

void initGLEvent(int argc, char** argv)
{
	//GL INITIALIZATION:
	glutInit(&argc, argv); //glutInit will initialize the GLUT library to operate with the Command Line
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(windowWidth, windowHeight);
	mainWindow = glutCreateWindow("OCT Viewer");
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

	fundusWindow = glutCreateSubWindow(mainWindow, 512,0,subWindWidth,subWindHeight);
	glutDisplayFunc(displayFundus);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glewInit();
	initFundusTexture();

	volumeWindow = glutCreateSubWindow(mainWindow, 0,512,subWindWidth,subWindHeight);
	glutDisplayFunc(displayVolume);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc (mouseWheel);
	glutMotionFunc(motion);
	glutReshapeFunc(resize);
	glewInit();
	initVolumeTexture();

	//End of GL callback functions


	//glutTimerFunc is a global callback function
	//Meaning it is not associated with any window
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
	//End of GL callback functions
//END OF GL INITIALIZATION

	if (volumeRender || fundusRender) {
		if (processData) {
			checkCudaErrors( cudaMalloc((void**)&d_volumeBuffer, width * height * frames * sizeof(float) ));
			cudaMemset( d_volumeBuffer, 0, width * height * frames * sizeof(float));
		} else if (!processData) {
			/************* These Functions have been modified from NVIDIA's volumeRender_kernel.cu at the following link: ************/
			/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#volume-rendering-with-3d-textures *********************/
			/**/copyVolumeToFloat();
			/**/cudaExtent volumeSize = make_cudaExtent(width, height, frames);
			/**/initRayCastCuda((void *)h_floatVolumeBuffer, volumeSize, cudaMemcpyHostToDevice);
			/*************************************************************************************************************************/
		}
	}

	/************* These Functions have been modified from NVIDIA's bilateral_kernel.cu at the following link: ***************/
	/************* http://docs.nvidia.com/cuda/cuda-samples/index.html#bilateral-filter **************************************/
	/**/updateGaussian(gaussian_delta, filter_radius);
	/*************************************************************************************************************************/

	printUserManual();
}

void runGLEvent() 
{
	glutMainLoopEvent();
}


void setBufferPtr( unsigned short *h_buffer)
{
	buffer1 = h_buffer;
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

void registerCudaHost ()
{
	cudaHostRegister(buffer1, width * height * frames * sizeof(unsigned short), cudaHostRegisterDefault);
}

void fundusSwitch()
{
	if (fundusRender) {
		size_t size;
		checkCudaErrors(cudaGraphicsMapResources(1,&fundusCudaRes,0));
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
		checkCudaErrors(cudaGraphicsMapResources(1,&volumeCudaRes,0));
		cudaGraphicsResourceGetMappedPointer((void**) &d_DisplayBuffer, &size, volumeCudaRes);
		checkCudaErrors(cudaMemset(d_DisplayBuffer, 0, size));
		cudaGraphicsUnmapResources(1,&volumeCudaRes,0);
		glutSetWindow(volumeWindow);
		glutPostRedisplay();
	}
	volumeRender = !volumeRender;
}


void initGLVarAndPtrs(bool procesData,
								 bool volumeRend,
								 bool fundRend,
								 int frameWid, 
								 int frameHei, 
								 int framesPerBuff,
								 int fileLength,
								 int winWid,
								 int winHei,
								 int volumeMode)
{
	cudaGLSetGLDevice(0);

	processData = procesData;
	volumeRender = volumeRend;
	fundusRender = fundRend;
	frameAveraging = false;
	bilatFilt = false;
	width = frameWid;
	height = frameHei;

	frames = fileLength/(frameWid*frameHei);

	if (frames%framesPerBuff != 0)
		frames -= framesPerBuff;
	if (processData) {
		framesPerBuffr = framesPerBuff;
	} else {
		framesPerBuffr = 1;
	}

	framesToAvg = 4;

	windowWidth = winWid;
	windowHeight = winHei;
	voxelThreshold = 0.020f;

	if (volumeRender) {
		displayMode = (volumeDisplay)volumeMode;
		cropOffset = 0;

		if (processData) {
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
		} else if (!processData) {
			h_floatVolumeBuffer = (float *)malloc(width * height * frames * sizeof(float));
			memset(h_floatVolumeBuffer, 0, width * height * frames * sizeof(float));
		}

		bscanWidth = volumeWidth;
		bscanHeight = volumeHeight;

	} else {
		if (!processData) {
			h_floatFrameBuffer = (float *)malloc(width * height * framesPerBuffr * sizeof(float));
			memset(h_floatFrameBuffer, 0, width * height * framesPerBuffr * sizeof(float));
		}
		bscanWidth = width;
		bscanHeight = height;
		volumeWidth = width;
		volumeHeight = height;
	}

	mainTexBuffer = (unsigned char *) malloc (mainTextureWidth*mainTextureHeight*3);
}