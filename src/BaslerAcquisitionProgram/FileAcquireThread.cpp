/**********************************************************************************
Filename	: FileAcquireThread.cpp
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

#include "FileAcquireThread.h"

void AcquireThreadFile::runFileAcquire()
{
	//	Simulate Data Acquisition
	unsigned short *DataFramePos;

    //Infinite Loop
	//Temporarily we have incorporated an infinite loop. Stop the program execution by closing the program. 
	//Threads can also be suspended and resumed via the start/stop buttons
	while (true)
	{
		DataFramePos = RawData;
		fread(DataFramePos, sizeof (short),globalOptions->IMAGEWIDTH * globalOptions->IMAGEHEIGHT * globalOptions->NumFramesPerVol, fid);
		if (feof(fid))
			rewind(fid);
	}
}

