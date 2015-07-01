/**********************************************************************************
Filename	: ProcessThread.cpp
Authors		: Kevin Wong, Yifan Jian, Jing Xu, Marinko Sarunic
Published	: March 14th, 2013

Copyright (C) 2012 Biomedical Optics Research Group - Simon Fraser University

This file is part of a free software. Details of this software has been described 
in the paper: 

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
#include "ProcessThread.h"
#include <math.h>


//Define Extern Prototypes from CUDA and OpenGL
/*************************************************************************************************************************/
extern "C" void initGLVarAndPtrs(bool procesData,
								 bool volumeRend,
								 bool fundRend,
								 bool slowBsc,
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
extern "C" void setFrameSize(int frameSize);
extern "C" void fundusSwitch();
extern "C" void volumeSwitch();
extern "C" void initGLEvent(int argc, char** argv);
extern "C" void runGLEvent();
/*************************************************************************************************************************/

void ProcessThread::runGPUProcess()
{
	unsigned short *DataFramePos;
	bool initCuda = false;
	bool registeredHost = false;
	const bool startFundus = true; //Modify this value to turn en face ON or OFF when bVolumeScan==true
	const bool startVolume = false; //Modify this value to turn volume rendering ON or OFF when bVolumeScan==true
	const bool slowBscan = false;//This option is currently used only for fundus + Bscan mode, and no volume
	//Slow Bscan requires more GPU memory than without slow Bscan, so ENSURE that the GPU has enough memory for the specified volume size
	//E.G. if Volume size is 2048x512x400, ensure GPU has AT LEAST 2048x512x400xsizeof(float)= 1.6GB of memory (perhaps even more)
	float mul = float(globalOptions->IMAGEWIDTH);

	//2nd, 3rd, and 4th order corrections for interpolation
	//The values (constant) for the second – further order corrections  are based on measurements for our particular system. 
	//For display purposes, we found that it was sufficient to adjust the value of lambda rather than the correction factors, 
		//while minimizing the complexity of the user interface.
	const float SECOND_ORDER_CORRECTION = 0.0f;
	const float THIRD_ORDER_CORRECTION	= 0.0f;
	const float FOURTH_ORDER_CORRECTION = 0.0f;
	float lambda_Max= globalOptions->LambdaMin	+	(mul-1) * globalOptions->dLambda	+
					pow((mul-1.0f),2) * SECOND_ORDER_CORRECTION	+
					pow((mul-1.0f),3) * THIRD_ORDER_CORRECTION	+
					pow((mul-1.0f),4) * FOURTH_ORDER_CORRECTION;

	float dispMag = 10.0f;
	int samplingMethod = 0;
	int fftLenMult = 2;
	bool fundusStatus = startFundus;
	bool volumeStatus = startVolume;
	int framesPerBuffer = 5;

	if (!initCuda) {
		initGLVarAndPtrs(	true,				//ProcessData, ProcessData false is currently deprecated in this version of the code
							startVolume,		//VolumeRender
							startFundus,		//Fundus Render
							slowBscan,
							globalOptions->IMAGEWIDTH, 
							globalOptions->IMAGEHEIGHT, 
							framesPerBuffer, 
							globalOptions->IMAGEWIDTH * globalOptions->IMAGEHEIGHT * globalOptions->NumFramesPerVol, 
							768, //Window Width, Subwindow Width is always half of this value
							768, //Window Height, Subwindow Height is always half of this value
							samplingMethod,
							1);

/* Initialize CUDA Variables and Pointers */
		initCudaProcVar(	globalOptions->IMAGEWIDTH, 
							globalOptions->IMAGEHEIGHT, 
							framesPerBuffer, 
							globalOptions->LambdaMin,
							lambda_Max,
							dispMag,
							0, //globalOptions->a2, for dispersion, use default value of 0 instead
							0, //globalOptions->a3, for dispersion, use default value of 0 instead
							samplingMethod,
							fftLenMult);
		initGLEvent(0, 0);
		initCuda = true;
	}

    //Infinite Loop
	//Temporarily we have incorporated an infinite loop. Stop the program execution by closing the program. 
	//Threads can also be suspended and resumed via the start/stop buttons
	while (true)
	{
		DataFramePos = RawData;
		setFrameSize(globalOptions->NumFramesPerVol);
		setBufferPtr(DataFramePos);
		if (!registeredHost) {
			//Comment out the registerCudaHost if too much host memory is being allocated
			//Registering too much page-locked memory could crash the program
			registerCudaHost();
			registeredHost = true;
		}
		
		//runGLEvent executes one iteration of the CUDA code, including:
		//-Processing
		//-fundus render aka sum voxel (optional)
		//-volume render aka raycast (optional)
		//-bilateral filtering (optional)
		//-averaging (optional)
		//-Display for line-plot, B-scan, fundus, and volume
		runGLEvent();

		if (startFundus && fundusStatus != globalOptions->bVolumeScan) {
			fundusStatus = globalOptions->bVolumeScan;
			fundusSwitch();
		}

		if (startVolume && volumeStatus != globalOptions->bVolumeScan) {
			volumeStatus = globalOptions->bVolumeScan;
			volumeSwitch();
		}
	} //while (true)
}
