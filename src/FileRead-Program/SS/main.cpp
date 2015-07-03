/**********************************************************************************
Filename	: main.cpp
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

#include "thread.hpp"
#include "FileReadThread.hpp"
#include "ProcessThread.hpp"
#include <iostream>

using namespace std;

int main()
{
	bool fileRead = true;

	int frameWidth = 1024;
	int frameHeight = 256;
	int framesPerBuffer = 20;
	int volumeSize = 256;


//OpenGL ONLY Variables to be transferred
	//Process Data MUST be true
	//Display pre-processed data currently does NOT work!!
	bool processData = true;  //MUST be true
	bool fundusRender = true; //Can be false or true
	bool volumeRender = true; //Can be false or true
	int fileLen;
	int bufferLen;
	int windowWidth = 1024;
	int windowHeight = 1024;
	buffer *h_buffer = new buffer[BUFFNUM];
	int buffCtr = 0;
	unsigned short *h_ProcBuffer;
	bool *registeredHost;
	enum volumeDisplay {DownSize, Crop};
	volumeDisplay volumeMode = Crop;

//CUDA ONLY Variables to be transferred
	int fftLenMult = 2;
// END OF DEFINING CUDA VARIABLES

	//Define the filename to be used for file acquisition simulation
	char *fileName = new char[100];
	fileName = "../../../.data/Mouse-Dataset.unp";

	FILE *file = fopen(fileName, "rb");
	if (file==NULL)
	{	printf("Unable to Open file. Exiting Program...\n"); exit(1);}
	fseek(file, 0, SEEK_END);
	fileLen=ftell(file)/(int)sizeof(unsigned short);
	fclose(file);
	
	if (volumeRender || fundusRender) {
		if (fileRead) {
			bufferLen = fileLen;
		} else {
			bufferLen = frameWidth*frameHeight*volumeSize;
		}
	} else {
		bufferLen = frameWidth*frameHeight*framesPerBuffer;
	}

	for (int i=0; i<BUFFNUM; i++) {
		h_buffer[i].data = (unsigned short *)malloc(bufferLen * sizeof(unsigned short));
		h_buffer[i].regHost = false;
	}


//Initiate FileRead Thread
	FileReadThread FileRead;
	FileRead.create();
	FileRead.InitFileRead(fileName, h_buffer, &buffCtr, &bufferLen);

//Initiate the Process Thread
	ProcessThread Proc;
	Proc.create();
	Proc.InitProcess(	h_buffer, &buffCtr, processData, volumeRender, fundusRender, 
						frameWidth, frameHeight, framesPerBuffer, bufferLen, 
						windowWidth, windowHeight, (int)volumeMode, fftLenMult);
//END OF INITIALIZATION
//Begin Process Thread
	FileRead.start();
	Proc.start();
	FileRead.join();
	Proc.join();

	return 0;
}


