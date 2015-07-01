/**********************************************************************************
Filename	: ScanningProcs.cpp
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

#include "ScanningProcs.h"

void ScanThreadLinear::InitializeSyncAndScan(void)
{
//CLEAR TASKS
	DAQmxClearTask(taskClock);
	DAQmxClearTask(taskTrig);
	DAQmxClearTask(taskTrA);
	DAQmxClearTask(taskAnalog);
	
	Samps = globalOptions->IMAGEHEIGHT+NumPtsDw;
	LineRate = Samps*FrameRate;
	
	//CREATE TASKS
	DAQmxCreateTask("",&taskAnalog);
	DAQmxCreateTask("",&taskTrig);
	DAQmxCreateTask("",&taskTrA);
	DAQmxCreateTask("",&taskClock);

	
	/*************************
		   CLOCK SOURCE 
	*************************/
	//CREATE INTERNAL CLOCK SOURCE
	DAQmxCreateCOPulseChanFreq(taskClock, "Dev1/ctr0", "", DAQmx_Val_Hz, 
		DAQmx_Val_Low, __DDELAY, FrameRate, .2);

	
	/*************************
		DIGITAL PULSE TRAIN
	*************************/
	//CREATE DIGITAL LINE
	DAQmxCreateDOChan(taskTrig,"Dev1/port0/line0","",DAQmx_Val_ChanPerLine);

	//SET TIMING AND STATE CLOCK SOURCE
	//Clock source is based off the analog sample clock
	DAQmxCfgSampClkTiming(taskTrig,"ao/SampleClock",FrameRate*Samps,
		DAQmx_Val_Rising,DAQmx_Val_ContSamps,Samps);
	
	/*************************
		ANALOG SAW TOOTH
	*************************/
	//CREATE ANALOG CHANNEL
	DAQmxCreateAOVoltageChan(taskAnalog,"Dev1/ao0",
			   "",-10,10.0,DAQmx_Val_Volts,NULL);
	DAQmxCreateAOVoltageChan(taskAnalog,"Dev1/ao1",
		   "",-10,10.0,DAQmx_Val_Volts,NULL);

	//SET TIMING
	DAQmxCfgSampClkTiming(taskAnalog,"",FrameRate*Samps,
		DAQmx_Val_Rising,DAQmx_Val_ContSamps,Samps);

	//GET TERMINAL NAME OF THE TRIGGER SOURCE
	GetTerminalNameWithDevPrefix(taskTrig, "Ctr0InternalOutput",trigName);
	
	//TRIGGER THE ANALOG DATA BASED OFF INTERNAL TRIGGER (CLOCK SOURCE)
	DAQmxCfgDigEdgeStartTrig(taskAnalog,trigName,DAQmx_Val_Rising);

	if (globalOptions->bVolumeScan == false)
	{
		GenSawTooth(Samps,XScanVolts_mV,XScanOffset_mV,ScanBuff);

		for (int i = 0; i< Samps; i++)
			VolBuff[i] = ScanBuff[i];

		for (int i = Samps; i < 2*Samps; i++)
			VolBuff[i] = YScanOffset_mV/1000;

		DAQmxWriteAnalogF64(taskAnalog, Samps, false ,10 ,DAQmx_Val_GroupByChannel, VolBuff,NULL,NULL);

		//GENERATE PULSE TRAIN TO TRIGGER CAMERA
		GenPulseTrain(Samps,digiBuff);
		DAQmxWriteDigitalLines(taskTrig,Samps,false,10.0,DAQmx_Val_GroupByChannel,digiBuff,NULL,NULL);
	}
	else
	{
		
		int frameCount;

		GenSawTooth(Samps,XScanVolts_mV,XScanOffset_mV,ScanBuff);
		for (frameCount = 0; frameCount < globalOptions->NumFramesPerVol; frameCount++)
		{
			
			for (int i = 0; i< Samps; i++)
				VolumeBuff[i+frameCount*Samps] = ScanBuff[i];
		}

		GenStairCase(Samps,globalOptions->NumFramesPerVol,YScanVolts_mV, YScanOffset_mV, tempBuff);

		for (int i = 0; i < Samps*globalOptions->NumFramesPerVol; i++)
			VolumeBuff[i + Samps*globalOptions->NumFramesPerVol] = tempBuff[i];

		DAQmxWriteAnalogF64(taskAnalog, Samps*globalOptions->NumFramesPerVol, false ,10 ,DAQmx_Val_GroupByChannel, VolumeBuff,NULL,NULL);

		//GENERATE PULSE TRAIN TO TRIGGER CAMERA
		for (int frameCount = 0; frameCount < globalOptions->NumFramesPerVol; frameCount++)
		{
			GenPulseTrain(Samps,digiBuff);
			for (int i = 0; i< Samps; i++)
				digiVolBuff[i+frameCount*Samps] = digiBuff[i];
		}

		DAQmxWriteDigitalLines(taskTrig,Samps*globalOptions->NumFramesPerVol,false,10.0,DAQmx_Val_GroupByChannel,digiVolBuff,NULL,NULL);
	}


	//GENERATE PULSE TRAIN TO TRIGGER CAMERA
	DAQmxCreateCOPulseChanFreq(taskTrA,"Dev1/ctr1","",DAQmx_Val_Hz,DAQmx_Val_Low,0.0,LineRate,0.2);
	DAQmxCfgImplicitTiming(taskTrA,DAQmx_Val_FiniteSamps,globalOptions->IMAGEHEIGHT);
	DAQmxCfgDigEdgeStartTrig(taskTrA,"/Dev1/PFI4",DAQmx_Val_Rising);
	DAQmxSetStartTrigRetriggerable(taskTrA, 1);
	DAQmxConnectTerms ("/Dev1/Ctr1InternalOutput", "/Dev1/RTSI0",  DAQmx_Val_DoNotInvertPolarity);
	
	//START TASKS
	//IMPORTANT - Need to arm analog task first to make sure that the digital and analog waveforms are in sync
	DAQmxStartTask(taskAnalog);
	DAQmxStartTask(taskTrA);
	DAQmxStartTask(taskTrig);
	DAQmxStartTask(taskClock);
}

void ScanThreadLinear::StopTasks(void)
{
	DAQmxStopTask(taskAnalog);
	DAQmxStopTask(taskTrig);
	DAQmxStopTask(taskTrA);
	DAQmxStopTask(taskClock);
	
	//CLEAR TASKS
	DAQmxClearTask(taskAnalog);
	DAQmxClearTask(taskTrig);
	DAQmxClearTask(taskTrA);
	DAQmxClearTask(taskClock);
}


void ScanThreadLinear::GenSawTooth(int numElements, double amplitude,  double offset, double sawTooth[])
{
	//All voltages are in millivolts. Dividing by 1000 converts millivolts to volts
	double stepSize = amplitude/(numElements-NumPtsDw-__DEADTIMEPTS-1)/1000;

	int i=0;
	for(i; i<__DEADTIMEPTS; i++)
		sawTooth[i] = (offset/1000 + amplitude/2000);

	for(i; i<numElements-NumPtsDw-__DEADTIMEPTS; i++)
		sawTooth[i] = (offset/1000 + amplitude/2000)- i*stepSize;
	
	double stepSize2 = amplitude/NumPtsDw/1000;
	int c = 0;
	for(i; i< (numElements - __DEADTIMEPTS); i++)
	{
		sawTooth[i] = sawTooth[numElements-NumPtsDw-__DEADTIMEPTS-1] + c*stepSize2;
		c++;
	}

	for(i; i< (numElements); i++)
		sawTooth[i] = offset/1000 + amplitude/2000;
}



void ScanThreadLinear::GenPulseTrain(int numElements, uInt8 digWave[])
{
	for(int i =0;i<numElements/2;i++)
		digWave[i] = (uInt8)(1);
	for(int i =numElements/2;i<numElements;i++)
		digWave[i] = (uInt8)(0);
	for (int i=0; i<__PULSEDELAY;i++)
		digWave[i] = (uInt8)(0);
}

void ScanThreadLinear::GenStairCase(int numElementsPerStep, int numSteps, double amplitude, double offset, double stairCase[])
{
	int stepdw = 3;
	double stepsize = amplitude/1000/(numSteps-stepdw-1); 
	int i=0;
	for (i = 0 ; i <numSteps-stepdw; i++)
	{
		for (int j = 0; j < numElementsPerStep; j++)
			stairCase[i*numElementsPerStep + j] = (offset/1000 - amplitude/2000) +  (i)*stepsize;
	}

	double stepsize_down = amplitude/1000/(numElementsPerStep*stepdw-1);
	for (int j = 0; j < numElementsPerStep*stepdw; j++)
		stairCase[i*numElementsPerStep+j] = (offset/1000 + amplitude/2000) -j*stepsize_down;
}


void ScanThreadLinear::GetTerminalNameWithDevPrefix(TaskHandle taskHandle, const char terminalName[], char triggerName[])
{
	long	error=0;
	char	device[256];
	long	productCategory;
	unsigned long	numDevices,i=1;

	DAQmxGetTaskNumDevices(taskHandle,&numDevices);
	while( i<=numDevices ) {
		DAQmxGetNthTaskDevice(taskHandle,i++,device,256);
		DAQmxGetDevProductCategory(device,&productCategory);
		if( productCategory!=DAQmx_Val_CSeriesModule && productCategory!=DAQmx_Val_SCXIModule ) {
			*triggerName++ = '/';
			strcat(strcat(strcpy(triggerName,device),"/"),terminalName);
			
			break;
		}
	}
}