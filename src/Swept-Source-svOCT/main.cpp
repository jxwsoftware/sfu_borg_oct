/**********************************************************************************
Filename	: main.cpp
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

#include "thread.hpp"
#include "FileReadThread.hpp"
#include "ProcessThread.hpp"
#include <iostream>

using namespace std;

// All the parameters are defined here. 
// Currently the code only works on a BM scan size of 3. (Hard coded)

int main()
{
	bool fileRead = true;

	int frameWidth =1024;
	int frameHeight =300; 
	int framesPerBuffer= 30;
	int framesTotal =900;

	//OpenGL ONLY Variables to be transferred
	//Process Data MUST be true
	//Display pre-processed data currently does NOT work!!
	bool processData = true;  //MUST be true
	bool fundusRender = true; //Can be false or true
	bool volumeRender = false; //Volume render currently does not work with speckle variance in this version of the code
	int bufferLen;
	int windowWidth = 1152;
	int windowHeight = 768;
	buffer *h_buffer = new buffer[BUFFNUM];
	int buffCtr = 0;
	enum volumeDisplay {DownSize, Crop};
	volumeDisplay volumeMode = Crop;

//CUDA ONLY Variables to be transferred
	int fftLenMult = 2;
// END OF DEFINING CUDA VARIABLES

	//Define the filename to be used for file acquisition simulation
	char *fileName = new char[100];
	//fileName ="D:\\ECC-SS-DATA\\06122013\\1000-5000\\jing-os-5000-2.unp";
	//fileName ="D:\\ECC-SS-DATA\\20112013\\od-big-2-5.unp";
	fileName ="G:\\kevin ECC Dec 27\\Mom\\3-od-fov-2-4.unp";
	//fileName = "D:\\ECC-SS-DATA\\02122013\\mei\\mei-od-1.unp";
	bufferLen = frameWidth*frameHeight*framesTotal;
	for (int i=0; i<BUFFNUM; i++) {
		h_buffer[i].data = (unsigned short *)malloc(bufferLen * sizeof(unsigned short));
		h_buffer[i].regHost = false;
	}
//Initiate FileRead Thread
	FileReadThread FileRead;
	FileRead.create();
	FileRead.InitFileRead(fileName, frameWidth, frameHeight, framesTotal, h_buffer, &buffCtr);

//Initiate the Process Thread
	ProcessThread Proc;
	Proc.create();
	Proc.InitProcess(	h_buffer, &buffCtr, processData, volumeRender, fundusRender, 
						frameWidth, frameHeight, framesPerBuffer, framesTotal, 
						windowWidth, windowHeight, (int)volumeMode, fftLenMult);


//END OF INITIALIZATION
//Begin Process Thread
	FileRead.start();
	Proc.start();
	FileRead.join();
	Proc.join();

	return 0;
}


