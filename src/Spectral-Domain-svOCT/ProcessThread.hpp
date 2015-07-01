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
#ifndef _PROCESS_THREAD_
#define _PROCESS_THREAD_

#include "thread.hpp"
#include "Global.hpp"
#include <iostream>


class ProcessThread : public Win32Thread
{
public:

	buffer *h_buff;
	int *buffCtr;

	bool processData;
	bool volumeRender;
	bool fundusRender;
	int frameWidth;
	int frameHeight;
	int framesPerBuffer;
	int framesTotal;
	int bufferLen;
	int windowWidth;
	int windowHeight;
	int samplingMethod;
	int volumeMode;
	int fftLenMult;

	float lambdaMin;
	float lambdaMax;
	float dispMag;
	float dispVal;
	float dispValThird;

	unsigned short *procBuffer;

	ProcessThread() : Win32Thread() {}
	~ProcessThread() {}

	void InitProcess(	buffer *h_buffer, int *buffCounter, bool procData, bool volRend, bool fundRend, 
						int frameWid, int frameHei, int framesPerBuff,int framesTot, int buffLen, 
						int winWid, int winHei, int sampMethod, int volMode, int fftLen,
						float lambda_Min, float lambda_Max, float disp_Mag, float disp_Val, float disp_ValThird);
	void cleanProcessThread();

private:
	void run();
};


#endif