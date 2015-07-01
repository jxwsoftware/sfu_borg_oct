/**********************************************************************************
Filename	: ProcessThread.cpp
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


#include <iostream>

#include "cuda_ProcHeader.cuh"
#include "ProcessThread.hpp"


//////////////////////////////////////////////////////////////////
void initGLVarAndPtrs(bool procesData,
								 bool volumeRend,
								 bool fundRend,
								 int frameWid, 
								 int frameHei, 
								 int framesPerBuff,
								 int fileLength,
								 int winWid,
								 int winHei,
								 int volumeMode);

void initCudaProcVar(	int frameWid, 
									int frameHei, 
									int framesPerBuff, 
									int fftLenMult);

void setBufferPtr( unsigned short *h_buffer);
void registerCudaHost();
void initGLEvent(int argc, char** argv);
void runGLEvent();
//////////////////////////////////////////////////////////////////



void ProcessThread::InitProcess(	buffer *h_buffer, int *buffCounter, bool procData, bool volRend, bool fundRend, 
									int frameWid, int frameHei, int framesPerBuff, 
									int buffLen, int winWid, int winHei, int volMode, int fftLen)
{
//Initalize Buffer-related items
	h_buff = h_buffer;
	buffCtr = buffCounter;

//Initialize all Process Thread variables, so that CUDA can be initialized and launched all within this thread
	processData = procData;
	volumeRender = volRend;
	fundusRender = fundRend;
	frameWidth = frameWid;
	frameHeight = frameHei;
	framesPerBuffer = framesPerBuff;
	bufferLen = buffLen;
	windowWidth = winWid;
	windowHeight = winHei;
	volumeMode = volMode;
	fftLenMult = fftLen;
}

void ProcessThread::run()
{
	//Initialize GL Parameters
	initGLVarAndPtrs(	processData, 
						volumeRender,
						fundusRender,
						frameWidth, 
						frameHeight, 
						framesPerBuffer, 
						bufferLen, 
						windowWidth, 
						windowHeight,
						volumeMode);


	//Initialize CUDA Parameters
	initCudaProcVar(frameWidth, 
					frameHeight, 
					framesPerBuffer, 
					fftLenMult);


	//Initialize Buffer into GL
	setBufferPtr(h_buff[0].data);
	initGLEvent(0, 0);

	int buffCounter;
	
	//Infinite Loop for Processing and Display
	while (true) 
	{
		buffCounter = *buffCtr;
		setBufferPtr(h_buff[buffCounter].data);
		if (!h_buff[buffCounter].regHost) {
			registerCudaHost();
			h_buff[buffCounter].regHost = true;
		}
		runGLEvent();
	}
}
