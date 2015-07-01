/**********************************************************************************
Filename	: FileReadThread.hpp
Authors		: Jing Xu, Kevin Wong, Yifan Jian, Marinko Sarunic
Published	: December 6th, 2012

Copyright (C) 2012 Biomedical Optics Research Group - Simon Fraser University
This software contains source code provided by NVIDIA Corporation.

This file is part of a Open Source software. Details of this software has been described 
in the papers titled: 

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

#include "FileReadThread.hpp"
#include <iostream>
#include <time.h>


void FileReadThread::InitFileRead(char *fileName, int frameWid, int frameHei, int desiredFrame, buffer *h_buffer, int *buffCounter)
{
	buffCtr = 0;
	buffCtr = buffCounter;
	h_buff = h_buffer;
	file = fopen(fileName, "rb");
	if (file==NULL)
	{
		printf("Unable to Open file. Exiting Program...\n");
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	fileLen=ftell(file)/(int)sizeof(unsigned short);
	rewind (file);
	bufferLen = fileLen;
	frameWidth = frameWid;
	frameHeight = frameHei;
	desiredFrames = desiredFrame;

	storageBuffer = (unsigned short *) malloc (frameWidth*frameHeight*desiredFrames * sizeof(unsigned short));
	fread(storageBuffer, sizeof(unsigned short), frameWidth*frameHeight*desiredFrames, file);
	printf("buffer..%d,frameWidth*frameHeight*desiredFrames....%d\n",bufferLen,frameWidth*frameHeight*desiredFrames);
	fclose(file);
}	

void FileReadThread::run()
{
	int curFileLoc = 0;
	int bufferCtr = 0;


	int oldFrameWidth = frameWidth;
	int oldFrameHeight = frameHeight;
	int totalFrames = desiredFrames;
	int widthOffset = 0;
	int heightOffset = 60;
	int frameOffset = 0;
	int incrementValue = 1;

	//clock_t t1, t2;

	int storageIdx = 0;
	int outputIdx = 0;
	int inputIdx = 0;


	while(true)
	{
		//t1 = clock();

		*buffCtr = bufferCtr;
		bufferCtr++;
		if (bufferCtr == BUFFNUM) {
			bufferCtr = 0;
		}

		//Acquisition Simulation by memcpy
		memcpy( h_buff[bufferCtr].data, 
			&storageBuffer[frameOffset*oldFrameWidth*oldFrameHeight], 	
			frameWidth*frameHeight*desiredFrames*sizeof(unsigned short));

		if (desiredFrames < totalFrames) {
			if (frameOffset==0) 
				incrementValue = 1;
			else if (frameOffset==(totalFrames-desiredFrames-1))
				incrementValue = (totalFrames-desiredFrames-1);

			frameOffset = (frameOffset+incrementValue) % (totalFrames-desiredFrames);
		}
	}

	fclose(file);

}