/**********************************************************************************
Filename	: FileReadThread.hpp
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

#include "FileReadThread.hpp"
#include <iostream>


void FileReadThread::InitFileRead(char *fileName, buffer *h_buffer, int *buffCounter, int *buffLen)
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

	if (*buffLen == NULL) {
		bufferLen = fileLen;
		*buffLen = fileLen;
	} else {
		bufferLen = *buffLen;
	}
}	

void FileReadThread::run()
{
	int curFileLoc = 0;
	int bufferCtr = 0;


	while(true)
	{
		*buffCtr = bufferCtr;

		bufferCtr++;
		if (bufferCtr == BUFFNUM) {
			bufferCtr = 0;
		}

		//Acquisition Simulation
		fread(h_buff[bufferCtr].data, 1, bufferLen * sizeof(unsigned short), file);

		curFileLoc = ftell(file);
		if (curFileLoc==fileLen*sizeof(unsigned short)) {
			rewind(file);
		}

		//The Sleep function forces the current thread to sleep for a specified amount of time
		//The integer value represents the number of milliseconds to sleep
		//The memory bandwidth for reading can range from 5GB/s to 20GB/s for typical RAM
		//Therefore an equivalent acquisition speed can be derived memory bandwidth speeds
		//In our experiment, the average fileread time is approximately 20ms per 1024x256x200 volume size
		//An extra 5ms delay produces an approximate 2Mhz A-scan acquisition rate 
		//This assumes bufferLen = 1024x256x200
		Sleep(5);
		//End of Acquisition Simulation
	}

	fclose(file);

}