/**********************************************************************************
Filename	: ScanningProcs.h
Authors		: Kevin Wong, Yifan Jian, Jing Xu, Mei Young, Marinko Sarunic
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

#ifndef __SCANLINEARPROC
#define __SCANLINEARPROC

#include <stdio.h>
#include <NIDAQmx.h>
#include "GlobalClass.h"
#include "thread.h"

#define __DEADTIMEPTS 0
#define __DDELAY 0
#define __PULSEDELAY 0
#define __YFrameNum 500
#define __MAXHEIGHT 2500


class ScanThreadLinear : public Win32Thread
{
public:
	ScanThreadLinear() : Win32Thread() {}
    ~ScanThreadLinear() {}

//public functions
	void InitializeSyncAndScan(void);
	void StopTasks(void);
	void GenSawTooth(int numElements, double amplitude,  double offset, double sawTooth[]);
	void GenStairCase(int numElementsPerStep, int numSteps, double amplitude, double offset,double stairCase[]);
	void GenPulseTrain(int numElements,uInt8 digWave[]);
	void GetTerminalNameWithDevPrefix(TaskHandle taskHandle, const char terminalName[], char triggerName[]);

//public variables
    TaskHandle  taskHandle;
	TaskHandle  taskAnalog;
	TaskHandle	taskClock;
	TaskHandle	taskTrig;
	TaskHandle  taskTrA;
   	   
	double XScanVolts_mV;
	double XScanOffset_mV;
	double YScanVolts_mV;
	double YScanOffset_mV;
	int FrameRate;
	int Samps;
	int NumPtsDw;
	int LineRate;

	GlobalClass *globalOptions;

	// these can be dynamically allocated arrays instead of statically allocated arrays
	char trigName[256];
	double ScanBuff[__MAXHEIGHT];
	double ScanBuffY[__MAXHEIGHT];
	double VolBuff[2*__MAXHEIGHT];
	unsigned char digiBuff[__MAXHEIGHT];
	double VolumeBuff[2*__MAXHEIGHT*__YFrameNum];
	double tempBuff[__MAXHEIGHT*__YFrameNum];
	unsigned char digiVolBuff[__MAXHEIGHT*__YFrameNum];

private:
    void run()
    {
		//Scanning Thread 'run' function is intentionally left blank
		//Possible plans in the future to actually use ScanningProcs as a thread
		//For now ScanningProcs is simply a set of procedures and scanning variables
	}
};

#endif 
