/**********************************************************************************
Filename	: Form.h
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

#pragma once

#include <afxcoll.h>
#include "GlobalClass.h"
#include "ProcessThread.h"
#include "FileAcquireThread.h"
#include "ScanningProcs.h"
#include "BaslerAcquireThread.h"

using namespace System::IO;

namespace CUDAOCT
{
using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

public __gc class Form1 : public System::Windows::Forms::Form
{

public:
	bool FILEREAD;
	bool BASLER;
	bool RunningThreads;
	unsigned short* RawDataPtr;
	GlobalClass *globalOptions;
	AcquireThreadFile *AcqFilePtr;
	ProcessThread *ProcThrPtr;
	ScanThreadLinear *ScanThrPtr;
	AcquireThreadBasler *AcqBaslerPtr;
	static int strbuffsize=100;
	char*	envVirVarStr;
	char*	sectionName;
	char*	defaultName;
	char*	stringName;
	char*   iniFile;

//Private Functions
private: System::Windows::Forms::Label*  ImageWidthLabel;
private: System::Windows::Forms::Label*  ImageHeightLabel;
private: System::Windows::Forms::Label*  BScanFrameRateLabel;
private: System::Windows::Forms::Label*  FlyBackPointsLabel;
private: System::Windows::Forms::Label*  FrameScanLabel;
private: System::Windows::Forms::Label*  XAmp_mV_Label;
private: System::Windows::Forms::Label*  XOffset_mV_Label;
private: System::Windows::Forms::Label*  AmplitudeLabel;
private: System::Windows::Forms::Label*  PathNameLabel;
private: System::Windows::Forms::Label*  OffsetLabel;
private: System::Windows::Forms::Label*  NumOfFramesLabel;
private: System::Windows::Forms::Label*  YOffset_mV_Label;
private: System::Windows::Forms::Label*  YAmp_mV_Label;
private: System::Windows::Forms::Label*  FileNameLabel;
private: System::Windows::Forms::Label*  dLambdaLabel;
private: System::Windows::Forms::Label*  LambdaMinLabel;
private: System::Windows::Forms::CheckBox*  NewFileCheckBox;
private: System::Windows::Forms::CheckBox*  YscanOffChckBx;
private: System::Windows::Forms::CheckBox*  consoleCheckBox;
private: System::Windows::Forms::OpenFileDialog*  openFileDialog1;
private: System::Windows::Forms::SaveFileDialog*  saveFileDialog2;
private: System::Windows::Forms::Button*  saveVolumeFlag;
private: System::Windows::Forms::Button*  StartThreadButton;
private: System::Windows::Forms::Button*  StopButton;
private: System::Windows::Forms::TextBox*  fnameTxtBx;
private: System::Windows::Forms::TextBox*  pnameTxtBx;
private: System::Windows::Forms::TextBox*  ImageWidthTxtBx;
private: System::Windows::Forms::TextBox*  ImageHeightTxtBx;
private: System::Windows::Forms::TextBox*  StatusTextBox;
private: System::Windows::Forms::ComboBox*  acquireSelectionComboBox;
private: System::Windows::Forms::GroupBox*  SavingGroupBox;
private: System::Windows::Forms::GroupBox*  ScanningGroupBox;
private: System::Windows::Forms::GroupBox*  LightSourceGroupBox;
private: System::Windows::Forms::GroupBox*  hotKeysGroupBox;
private: System::Windows::Forms::ListBox*  hotKeyMenu;
private: System::Windows::Forms::NumericUpDown*  YFrameNumNumeric;
private: System::Windows::Forms::NumericUpDown*  LineRateNumeric;
private: System::Windows::Forms::NumericUpDown*  YscanOffset_mVNumeric;
private: System::Windows::Forms::NumericUpDown*  XScanAmp_mVNumeric;
private: System::Windows::Forms::NumericUpDown*  XScanOffset_mVNumeric;
private: System::Windows::Forms::NumericUpDown*  YscanAmp_mVNumeric;
private: System::Windows::Forms::NumericUpDown*  FlyBackPoints;
private: System::Windows::Forms::NumericUpDown*  LambdaMinNumeric;
private: System::Windows::Forms::NumericUpDown*  dLambdaNumeric;
private: System::Windows::Forms::TabControl*  allTabs;
private: System::Windows::Forms::TabPage*  OCTMainContTab;


public:
Form1(void)
{
	InitializeComponent();

	AllocConsole();
	SetConsoleTitle("Debug Console");
	freopen("CONOUT$", "wb", stdout);
	ShowWindow( GetConsoleWindow(), SW_HIDE);
	
	envVirVarStr = new char[strbuffsize];
	 //Be sure to set the environment variable called OCTVIEWERPATH to OCTviewer path, e.g. 'C:\OCTViewer'
	GetEnvironmentVariable("OCTVIEWERPATH",envVirVarStr,strbuffsize);
	iniFile = strcat(envVirVarStr, "\\OCTViewer_Basler.ini");
	sectionName = new char[strbuffsize];
	defaultName = new char[strbuffsize];
	stringName = new char[strbuffsize];

	_tcscpy(sectionName, _T("Acquisition"));	
	GetPrivateProfileString(sectionName, _T("ImageHeight"), defaultName, stringName, strbuffsize, iniFile);
	ImageHeightTxtBx->Text = stringName;
	GetPrivateProfileString(sectionName, _T("ImageWidth"), defaultName, stringName, strbuffsize, iniFile);
	ImageWidthTxtBx->Text = stringName;
	GetPrivateProfileString(sectionName, _T("NumberOfFrames"), defaultName, stringName, strbuffsize, iniFile);
	YFrameNumNumeric->Text = stringName;

	_tcscpy(sectionName, _T("Scanning"));	
	GetPrivateProfileString(sectionName, _T("XOffset"), defaultName, stringName, strbuffsize, iniFile);
	XScanOffset_mVNumeric->Text = stringName;
	GetPrivateProfileString(sectionName, _T("XScanRange"), defaultName, stringName, strbuffsize, iniFile);
	XScanAmp_mVNumeric->Text = stringName;
	GetPrivateProfileString(sectionName, _T("YOffset"), defaultName, stringName, strbuffsize, iniFile);
	YscanOffset_mVNumeric->Text = stringName;
	GetPrivateProfileString(sectionName, _T("YScanRange"), defaultName, stringName, strbuffsize, iniFile);
	YscanAmp_mVNumeric->Text = stringName;
	GetPrivateProfileString(sectionName, _T("FlyBackPoints"), defaultName, stringName, strbuffsize, iniFile);
	FlyBackPoints->Text = stringName;
	GetPrivateProfileString(sectionName, _T("BScanFrameRate"), defaultName, stringName, strbuffsize, iniFile);
	LineRateNumeric->Text = stringName;
	RunningThreads = false;

}//	Form1(void)


void InitializeThreads(void)
{
	globalOptions = new GlobalClass;
	ProcThrPtr = new ProcessThread;
	ProcThrPtr->create();
	ScanThrPtr = new ScanThreadLinear;
	ScanThrPtr->create();

	globalOptions->LambdaMin = Convert::ToSingle(LambdaMinNumeric->Value);
	globalOptions->dLambda = Convert::ToSingle(dLambdaNumeric->Value);
	globalOptions->bVolumeScan = YscanOffChckBx->Checked;
	globalOptions->saveVolumeflag = false;

	if (acquireSelectionComboBox->SelectedIndex == 0)
		BASLER = true;
	else 
		FILEREAD = true;

	if (FILEREAD) {
		AcqFilePtr = new AcquireThreadFile;
		AcqFilePtr->create();
		globalOptions->bVolumeScan = true; //If File read, then always display entire volume
	}
	else if (BASLER) {
		AcqBaslerPtr = new AcquireThreadBasler;
		AcqBaslerPtr->create();
		AcqBaslerPtr->ScanThrPtr = ScanThrPtr;
	}	

	LambdaMinNumeric->Enabled = false;
	dLambdaNumeric->Enabled = false;

	_tcscpy(sectionName, _T("Acquisition"));	
	
	globalOptions->IMAGEWIDTH = Convert::ToInt32(ImageWidthTxtBx->Text);
	ImageWidthTxtBx->Enabled = false;
	_itoa_s(globalOptions->IMAGEWIDTH,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("ImageWidth"),stringName,iniFile);
	
	globalOptions->IMAGEHEIGHT = Convert::ToInt32(ImageHeightTxtBx->Text);	
	ImageHeightTxtBx->Enabled = false;
	_itoa_s(globalOptions->IMAGEHEIGHT,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("ImageHeight"),stringName,iniFile);
		
	globalOptions->NumFramesPerVol = Convert::ToInt32(YFrameNumNumeric->Text);	
	_itoa_s(globalOptions->NumFramesPerVol,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("NumberOfFrames"),stringName,iniFile);
	
// initialize the scan offset and scan size from values saved on file
	_tcscpy(sectionName, _T("Scanning"));		
	
	ScanThrPtr->XScanOffset_mV = Convert::ToDouble(XScanOffset_mVNumeric->Text);
	_itoa_s(ScanThrPtr->XScanOffset_mV,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("XOffset"),stringName,iniFile);

	ScanThrPtr->XScanVolts_mV = Convert::ToDouble(XScanAmp_mVNumeric->Text);
	_itoa_s(ScanThrPtr->XScanVolts_mV,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("XScanRange"),stringName,iniFile);

	ScanThrPtr->YScanOffset_mV = Convert::ToDouble(YscanOffset_mVNumeric->Text);
	_itoa_s(ScanThrPtr->YScanOffset_mV,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("YOffset"),stringName,iniFile);
	
	ScanThrPtr->YScanVolts_mV = Convert::ToDouble(YscanAmp_mVNumeric->Text);
	_itoa_s(ScanThrPtr->YScanVolts_mV,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("YScanRange"),stringName,iniFile);

	ScanThrPtr->NumPtsDw = Convert::ToInt32(FlyBackPoints->Text);
	_itoa_s(ScanThrPtr->NumPtsDw,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("FlyBackPoints"),stringName,iniFile);
	
	ScanThrPtr->FrameRate = Convert::ToInt32(LineRateNumeric->Text);		
	_itoa_s(ScanThrPtr->FrameRate,stringName,strbuffsize,10);			
	WritePrivateProfileString(sectionName,_T("BScanFrameRate"),stringName,iniFile);

	//Memory Allocation for the main buffer
	RawDataPtr = (unsigned short * )malloc(globalOptions->NumFramesPerVol*globalOptions->IMAGEWIDTH*globalOptions->IMAGEHEIGHT*sizeof(unsigned short));
	
	if (NULL == RawDataPtr)
	{
		StatusTextBox->Text = "Unable to allocate RawDataPtr ";
		StatusTextBox->Update();
		::Sleep(2000);
	}

	ScanThrPtr->globalOptions = globalOptions;

	//Setup global options
	globalOptions->volCount = 0;
	globalOptions->volumeReady = 0;

	//Setup the Processing thread	
	ProcThrPtr->RawData = RawDataPtr;
	ProcThrPtr->globalOptions = globalOptions;
		
	if (FILEREAD) //load fileread acquisition
	{
		StatusTextBox->Text = "Using FILE DATA";
		StatusTextBox->Update();
		//Setup the File Acquisition Thread (created in frozen state)
		AcqFilePtr->globalOptions = globalOptions;
		AcqFilePtr->RawData = RawDataPtr;
	}
	else if (BASLER)
	{
		StatusTextBox->Text = "Using BASLER";
		StatusTextBox->Update();
		::Sleep(500);
		//Setup the BASLER Acquisition Thread (created in frozen state)

		AcqBaslerPtr->RawData = RawDataPtr;
		AcqBaslerPtr->globalOptions = globalOptions;
	}
}//END ConfigAllParams
//End of all Public variables and functions

//Protected Functions
protected:
void Dispose(Boolean disposing)
{
	if (disposing && components)
	{
		components->Dispose();
	}
	__super::Dispose(disposing);
}


private: 
System::Void StartThreadButton_Click(System::Object *  sender, System::EventArgs *  e)
{
	if (!RunningThreads) //Start Running threads
	{
		StatusTextBox->Text = "Starting new thread";
		StatusTextBox->Update();
		::Sleep(200);
		InitializeThreads();

		RunningThreads = true;
		
		// start up the threads
		ProcThrPtr->start();

		//start the acquire thread
		if (FILEREAD)
		{
			if (openFileDialog1->ShowDialog() == DialogResult::OK )	
			{
				CString filename = openFileDialog1->FileName;
				int index=filename.Find(92,1);
				while(index!=-1)
				{
					filename.Insert(index,char(92));
					index=filename.Find(92,index+2);
				}
				StatusTextBox->Text = Convert::ToString(filename);
				fopen_s(&AcqFilePtr->fid,filename,"rb+");
				fseek(AcqFilePtr->fid,0,2);
				StatusTextBox->Update();
				AcqFilePtr->start();
			}

			StatusTextBox->Text = "File Read Thread STARTED";
			StatusTextBox->Update();
		}
		else if (BASLER)
		{
			AcqBaslerPtr->start();
			StatusTextBox->Text = "Basler Thread STARTED";
			StatusTextBox->Update();
		}

		ScanThrPtr->InitializeSyncAndScan();
	}
	else //Resume Threads from suspend state
	{
		StatusTextBox->Text = "Resume old thread";
		StatusTextBox->Update();
		ProcThrPtr->resume();
		
		if (FILEREAD)
		{
			AcqFilePtr->resume();

			if (NewFileCheckBox->Checked == true )
			{
				//read in a new file
				if (openFileDialog1->ShowDialog() == DialogResult::OK )	
				{
					FILE *newFid;
					FILE *oldFid;
					CString filename = openFileDialog1->FileName;
					int index=filename.Find(92,1);
					while(index!=-1)
					{
						filename.Insert(index,char(92));
						index=filename.Find(92,index+2);
					}
					StatusTextBox->Text = Convert::ToString(filename);

					oldFid = AcqFilePtr->fid;
					fopen_s(&newFid, filename,"rb+");
					AcqFilePtr->fid = newFid;
					fseek(AcqFilePtr->fid,0,2);
					fclose(oldFid);

					StatusTextBox->Update();
					NewFileCheckBox->Checked = false;
					NewFileCheckBox->Update();
				}		
			}			
		}
		else if (BASLER)
		{
			AcqBaslerPtr->resume();
			AcqBaslerPtr->breakLoop = true;
		}

		ScanThrPtr->resume();
		ScanThrPtr->InitializeSyncAndScan();
	}
}


private: 
System::Void StopButton_Click(System::Object *  sender, System::EventArgs *  e)
{
	if (RunningThreads) { 
		//Put acquisition threads into suspend state
		//Leave the Processing and Scanning threads in their current state
		//Both Processing and Scanning can remain running
		StatusTextBox->Text = "suspend the threads";

		if (FILEREAD)
			AcqFilePtr->suspend();
		else if (BASLER)
			AcqBaslerPtr->suspend();
	}
}


private: 
System::Void ScanValChanged (System::Object*  sender, System::EventArgs*  e) 
{
	if (ScanThrPtr!= NULL)
	{
		ScanThrPtr->NumPtsDw = Convert::ToInt32(FlyBackPoints->Text);
		globalOptions->bVolumeScan = YscanOffChckBx->Checked;
		ScanThrPtr->FrameRate = Convert::ToInt32(LineRateNumeric->Value);
		ScanThrPtr->XScanVolts_mV = Convert::ToDouble(XScanAmp_mVNumeric->Value);
		ScanThrPtr->XScanOffset_mV = Convert::ToDouble(XScanOffset_mVNumeric->Value);
		ScanThrPtr->YScanVolts_mV = Convert::ToDouble(YscanAmp_mVNumeric->Value);
		ScanThrPtr->YScanOffset_mV = Convert::ToDouble(YscanOffset_mVNumeric->Value);
		globalOptions->NumFramesPerVol = Convert::ToInt32(YFrameNumNumeric->Value);
		
		if (BASLER) {
			AcqBaslerPtr->breakLoop = YscanOffChckBx->Checked;
		}			
		ScanThrPtr->InitializeSyncAndScan();	

	}
}

private: 
System::Void saveVolumeFlag_Click(System::Object*  sender, System::EventArgs*  e) 
{
	if (RunningThreads)
	{	 			  	 
		char filename[500];
		sprintf_s(filename,  "%s%s-%d.unp",pnameTxtBx->Text,fnameTxtBx->Text,globalOptions->volCount++);
		
		fopen_s(&globalOptions->fid,filename,"wb+");
  
		if (FILEREAD == false)
			globalOptions->saveVolumeflag = true;
		else
			globalOptions->volumeReady = 0;

		char ltemp[1000];
		_tcscpy(ltemp, filename);
		strcat_s(ltemp, ".txt");
		fopen_s(&globalOptions->fid2,ltemp,"w+");

		char tempString[100];
		_itoa_s(globalOptions->IMAGEHEIGHT,tempString,10);
		fprintf(globalOptions->fid2,"Image Height: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(globalOptions->IMAGEWIDTH,tempString,10);
		fprintf(globalOptions->fid2,"\nImage Width: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(XScanAmp_mVNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nXScan Amplitude: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(YscanAmp_mVNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nYScan Volts: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(XScanOffset_mVNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nScan Offset: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(YscanOffset_mVNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nYScan Offset: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(YFrameNumNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nNumber of Frames: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(FlyBackPoints->Text),tempString,10);
		fprintf(globalOptions->fid2,"\nFlyback points: ");
		fprintf(globalOptions->fid2,tempString);
		_itoa_s(Convert::ToInt16(LineRateNumeric->Value),tempString,10);
		fprintf(globalOptions->fid2,"\nBscanFrameRate: ");
		fprintf(globalOptions->fid2,tempString);

		fclose(globalOptions->fid2);
		 			  			  
	}
}

private: 
System::Void fnameTxtBx_TextChanged(System::Object*  sender, System::EventArgs*  e) 
{	
	if (RunningThreads) 
		globalOptions->volCount = 1;
}

private: 
System::Void consoleCheckBox_CheckedChanged(System::Object*  sender, System::EventArgs*  e)
{
	if (consoleCheckBox->Checked) {
		ShowWindow( GetConsoleWindow(), SW_SHOW);
	} else {
		ShowWindow( GetConsoleWindow(), SW_HIDE);
	}
}

private:
System::ComponentModel::Container * components;
void InitializeComponent(void)
{
	this->StatusTextBox = (new System::Windows::Forms::TextBox());
	this->StopButton = (new System::Windows::Forms::Button());
	this->StartThreadButton = (new System::Windows::Forms::Button());
	this->acquireSelectionComboBox = (new System::Windows::Forms::ComboBox());
	this->openFileDialog1 = (new System::Windows::Forms::OpenFileDialog());
	this->saveFileDialog2 = (new System::Windows::Forms::SaveFileDialog());
	this->NumOfFramesLabel = (new System::Windows::Forms::Label());
	this->YFrameNumNumeric = (new System::Windows::Forms::NumericUpDown());
	this->NewFileCheckBox = (new System::Windows::Forms::CheckBox());
	this->saveVolumeFlag = (new System::Windows::Forms::Button());
	this->SavingGroupBox = (new System::Windows::Forms::GroupBox());
	this->FileNameLabel = (new System::Windows::Forms::Label());
	this->PathNameLabel = (new System::Windows::Forms::Label());
	this->pnameTxtBx = (new System::Windows::Forms::TextBox());
	this->fnameTxtBx = (new System::Windows::Forms::TextBox());
	this->consoleCheckBox = (new System::Windows::Forms::CheckBox());
	this->OCTMainContTab = (new System::Windows::Forms::TabPage());
	this->LightSourceGroupBox = (new System::Windows::Forms::GroupBox());
	this->LambdaMinNumeric = (new System::Windows::Forms::NumericUpDown());
	this->LambdaMinLabel = (new System::Windows::Forms::Label());
	this->dLambdaLabel = (new System::Windows::Forms::Label());
	this->dLambdaNumeric = (new System::Windows::Forms::NumericUpDown());
	this->ImageWidthTxtBx = (new System::Windows::Forms::TextBox());
	this->ImageHeightTxtBx = (new System::Windows::Forms::TextBox());
	this->ImageWidthLabel = (new System::Windows::Forms::Label());
	this->ScanningGroupBox = (new System::Windows::Forms::GroupBox());
	this->FlyBackPoints = (new System::Windows::Forms::NumericUpDown());
	this->FlyBackPointsLabel = (new System::Windows::Forms::Label());
	this->LineRateNumeric = (new System::Windows::Forms::NumericUpDown());
	this->YscanOffset_mVNumeric = (new System::Windows::Forms::NumericUpDown());
	this->BScanFrameRateLabel = (new System::Windows::Forms::Label());
	this->XScanAmp_mVNumeric = (new System::Windows::Forms::NumericUpDown());
	this->FrameScanLabel = (new System::Windows::Forms::Label());
	this->XAmp_mV_Label = (new System::Windows::Forms::Label());
	this->XScanOffset_mVNumeric = (new System::Windows::Forms::NumericUpDown());
	this->AmplitudeLabel = (new System::Windows::Forms::Label());
	this->XOffset_mV_Label = (new System::Windows::Forms::Label());
	this->YscanAmp_mVNumeric = (new System::Windows::Forms::NumericUpDown());
	this->OffsetLabel = (new System::Windows::Forms::Label());
	this->YOffset_mV_Label = (new System::Windows::Forms::Label());
	this->YAmp_mV_Label = (new System::Windows::Forms::Label());
	this->YscanOffChckBx = (new System::Windows::Forms::CheckBox());
	this->ImageHeightLabel = (new System::Windows::Forms::Label());
	this->allTabs = (new System::Windows::Forms::TabControl());
	this->hotKeysGroupBox = (new System::Windows::Forms::GroupBox());
	this->hotKeyMenu = (new System::Windows::Forms::ListBox());
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YFrameNumNumeric))->BeginInit();
	this->SavingGroupBox->SuspendLayout();
	this->OCTMainContTab->SuspendLayout();
	this->LightSourceGroupBox->SuspendLayout();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->LambdaMinNumeric))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->dLambdaNumeric))->BeginInit();
	this->ScanningGroupBox->SuspendLayout();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->FlyBackPoints))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->LineRateNumeric))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YscanOffset_mVNumeric))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->XScanAmp_mVNumeric))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->XScanOffset_mVNumeric))->BeginInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YscanAmp_mVNumeric))->BeginInit();
	this->allTabs->SuspendLayout();
	this->hotKeysGroupBox->SuspendLayout();
	this->SuspendLayout();
	// 
	// StatusTextBox
	// 
	this->StatusTextBox->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->StatusTextBox->Location = System::Drawing::Point(20, 411);
	this->StatusTextBox->Name = S"StatusTextBox";
	this->StatusTextBox->Size = System::Drawing::Size(366, 21);
	this->StatusTextBox->TabIndex = 2;
	this->StatusTextBox->Text = S"Status OK";
	// 
	// StopButton
	// 
	this->StopButton->BackColor = System::Drawing::SystemColors::ButtonFace;
	this->StopButton->FlatStyle = System::Windows::Forms::FlatStyle::System;
	this->StopButton->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->StopButton->ForeColor = System::Drawing::Color::Red;
	this->StopButton->Location = System::Drawing::Point(192, 5);
	this->StopButton->Name = S"StopButton";
	this->StopButton->Size = System::Drawing::Size(63, 29);
	this->StopButton->TabIndex = 7;
	this->StopButton->Text = S"Stop";
	this->StopButton->UseVisualStyleBackColor = false;
	this->StopButton->Click += new System::EventHandler(this, &Form1::StopButton_Click);
	// 
	// StartThreadButton
	// 
	this->StartThreadButton->BackColor = System::Drawing::SystemColors::ButtonFace;
	this->StartThreadButton->FlatStyle = System::Windows::Forms::FlatStyle::System;
	this->StartThreadButton->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->StartThreadButton->ForeColor = System::Drawing::Color::FromArgb((System::Int32)(System::Byte)0, (System::Int32)(System::Byte)192, 
		(System::Int32)(System::Byte)0);
	this->StartThreadButton->Location = System::Drawing::Point(126, 5);
	this->StartThreadButton->Name = S"StartThreadButton";
	this->StartThreadButton->Size = System::Drawing::Size(60, 29);
	this->StartThreadButton->TabIndex = 12;
	this->StartThreadButton->Text = S"Start";
	this->StartThreadButton->UseVisualStyleBackColor = false;
	this->StartThreadButton->Click += new System::EventHandler(this, &Form1::StartThreadButton_Click);
	// 
	// acquireSelectionComboBox
	// 
	this->acquireSelectionComboBox->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, 
		System::Drawing::GraphicsUnit::Point, (System::Byte)0));
	this->acquireSelectionComboBox->FormattingEnabled = true;
	System::Object* __mcTemp__1[] = new System::Object*[2];
	__mcTemp__1[0] = S"Basler";
	__mcTemp__1[1] = S"File Load";
	this->acquireSelectionComboBox->Items->AddRange(__mcTemp__1);
	this->acquireSelectionComboBox->Location = System::Drawing::Point(20, 8);
	this->acquireSelectionComboBox->Name = S"acquireSelectionComboBox";
	this->acquireSelectionComboBox->Size = System::Drawing::Size(91, 24);
	this->acquireSelectionComboBox->TabIndex = 44;
	this->acquireSelectionComboBox->Text = S"Basler";
	// 
	// openFileDialog1
	// 
	this->openFileDialog1->FileName = S"openFileDialog1";
	// 
	// NumOfFramesLabel
	// 
	this->NumOfFramesLabel->Location = System::Drawing::Point(269, 99);
	this->NumOfFramesLabel->Name = S"NumOfFramesLabel";
	this->NumOfFramesLabel->Size = System::Drawing::Size(76, 17);
	this->NumOfFramesLabel->TabIndex = 77;
	this->NumOfFramesLabel->Text = S"# of Frames";
	// 
	// YFrameNumNumeric
	// 
	this->YFrameNumNumeric->Location = System::Drawing::Point(269, 117);
	System::Int32 __mcTemp__2[] = new System::Int32[4];
	__mcTemp__2[0] = 1024;
	__mcTemp__2[1] = 0;
	__mcTemp__2[2] = 0;
	__mcTemp__2[3] = 0;
	this->YFrameNumNumeric->Maximum = System::Decimal(__mcTemp__2);
	this->YFrameNumNumeric->Name = S"YFrameNumNumeric";
	this->YFrameNumNumeric->Size = System::Drawing::Size(59, 21);
	this->YFrameNumNumeric->TabIndex = 78;
	System::Int32 __mcTemp__3[] = new System::Int32[4];
	__mcTemp__3[0] = 400;
	__mcTemp__3[1] = 0;
	__mcTemp__3[2] = 0;
	__mcTemp__3[3] = 0;
	this->YFrameNumNumeric->Value = System::Decimal(__mcTemp__3);
	this->YFrameNumNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// NewFileCheckBox
	// 
	this->NewFileCheckBox->AutoSize = true;
	this->NewFileCheckBox->Location = System::Drawing::Point(20, 37);
	this->NewFileCheckBox->Name = S"NewFileCheckBox";
	this->NewFileCheckBox->Size = System::Drawing::Size(210, 20);
	this->NewFileCheckBox->TabIndex = 84;
	this->NewFileCheckBox->Text = S"Load New File (File Load Only)";
	this->NewFileCheckBox->UseVisualStyleBackColor = true;
	// 
	// saveVolumeFlag
	// 
	this->saveVolumeFlag->FlatStyle = System::Windows::Forms::FlatStyle::System;
	this->saveVolumeFlag->Location = System::Drawing::Point(11, 19);
	this->saveVolumeFlag->Name = S"saveVolumeFlag";
	this->saveVolumeFlag->Size = System::Drawing::Size(151, 45);
	this->saveVolumeFlag->TabIndex = 95;
	this->saveVolumeFlag->Text = S"ACQUIRE";
	this->saveVolumeFlag->UseVisualStyleBackColor = true;
	this->saveVolumeFlag->Click += new System::EventHandler(this, &Form1::saveVolumeFlag_Click);
	// 
	// SavingGroupBox
	// 
	this->SavingGroupBox->BackColor = System::Drawing::Color::Transparent;
	this->SavingGroupBox->Controls->Add(this->FileNameLabel);
	this->SavingGroupBox->Controls->Add(this->PathNameLabel);
	this->SavingGroupBox->Controls->Add(this->pnameTxtBx);
	this->SavingGroupBox->Controls->Add(this->fnameTxtBx);
	this->SavingGroupBox->Controls->Add(this->saveVolumeFlag);
	this->SavingGroupBox->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->SavingGroupBox->Location = System::Drawing::Point(20, 335);
	this->SavingGroupBox->Name = S"SavingGroupBox";
	this->SavingGroupBox->Size = System::Drawing::Size(366, 70);
	this->SavingGroupBox->TabIndex = 107;
	this->SavingGroupBox->TabStop = false;
	this->SavingGroupBox->Text = S"Saving";
	// 
	// FileNameLabel
	// 
	this->FileNameLabel->AutoSize = true;
	this->FileNameLabel->Location = System::Drawing::Point(276, 13);
	this->FileNameLabel->Name = S"FileNameLabel";
	this->FileNameLabel->Size = System::Drawing::Size(64, 15);
	this->FileNameLabel->TabIndex = 99;
	this->FileNameLabel->Text = S"File Name";
	// 
	// PathNameLabel
	// 
	this->PathNameLabel->AutoSize = true;
	this->PathNameLabel->Location = System::Drawing::Point(172, 13);
	this->PathNameLabel->Name = S"PathNameLabel";
	this->PathNameLabel->Size = System::Drawing::Size(69, 15);
	this->PathNameLabel->TabIndex = 98;
	this->PathNameLabel->Text = S"Path Name";
	// 
	// pnameTxtBx
	// 
	this->pnameTxtBx->Location = System::Drawing::Point(173, 31);
	this->pnameTxtBx->Name = S"pnameTxtBx";
	this->pnameTxtBx->Size = System::Drawing::Size(98, 21);
	this->pnameTxtBx->TabIndex = 97;
	this->pnameTxtBx->Text = S"C:\\";
	// 
	// fnameTxtBx
	// 
	this->fnameTxtBx->Location = System::Drawing::Point(277, 31);
	this->fnameTxtBx->Name = S"fnameTxtBx";
	this->fnameTxtBx->Size = System::Drawing::Size(85, 21);
	this->fnameTxtBx->TabIndex = 96;
	this->fnameTxtBx->TextChanged += new System::EventHandler(this, &Form1::fnameTxtBx_TextChanged);
	// 
	// consoleCheckBox
	// 
	this->consoleCheckBox->Anchor = (System::Windows::Forms::AnchorStyles)(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
		| System::Windows::Forms::AnchorStyles::Left) 
		| System::Windows::Forms::AnchorStyles::Right);
	this->consoleCheckBox->AutoSize = true;
	this->consoleCheckBox->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9));
	this->consoleCheckBox->Location = System::Drawing::Point(269, 195);
	this->consoleCheckBox->Name = S"consoleCheckBox";
	this->consoleCheckBox->Size = System::Drawing::Size(71, 19);
	this->consoleCheckBox->TabIndex = 119;
	this->consoleCheckBox->Text = S"Console";
	this->consoleCheckBox->TextAlign = System::Drawing::ContentAlignment::TopLeft;
	this->consoleCheckBox->UseVisualStyleBackColor = true;
	this->consoleCheckBox->CheckedChanged += new System::EventHandler(this, &Form1::consoleCheckBox_CheckedChanged);
	// 
	// OCTMainContTab
	// 
	this->OCTMainContTab->BackColor = System::Drawing::Color::Transparent;
	this->OCTMainContTab->Controls->Add(this->consoleCheckBox);
	this->OCTMainContTab->Controls->Add(this->LightSourceGroupBox);
	this->OCTMainContTab->Controls->Add(this->ImageWidthTxtBx);
	this->OCTMainContTab->Controls->Add(this->ImageHeightTxtBx);
	this->OCTMainContTab->Controls->Add(this->YFrameNumNumeric);
	this->OCTMainContTab->Controls->Add(this->NumOfFramesLabel);
	this->OCTMainContTab->Controls->Add(this->ImageWidthLabel);
	this->OCTMainContTab->Controls->Add(this->ScanningGroupBox);
	this->OCTMainContTab->Controls->Add(this->ImageHeightLabel);
	this->OCTMainContTab->Location = System::Drawing::Point(4, 24);
	this->OCTMainContTab->Name = S"OCTMainContTab";
	this->OCTMainContTab->Padding = System::Windows::Forms::Padding(3);
	this->OCTMainContTab->Size = System::Drawing::Size(358, 242);
	this->OCTMainContTab->TabIndex = 0;
	this->OCTMainContTab->Text = S"OCT Main Controls";
	this->OCTMainContTab->UseVisualStyleBackColor = true;
	// 
	// LightSourceGroupBox
	// 
	this->LightSourceGroupBox->BackColor = System::Drawing::Color::Transparent;
	this->LightSourceGroupBox->Controls->Add(this->LambdaMinNumeric);
	this->LightSourceGroupBox->Controls->Add(this->LambdaMinLabel);
	this->LightSourceGroupBox->Controls->Add(this->dLambdaLabel);
	this->LightSourceGroupBox->Controls->Add(this->dLambdaNumeric);
	this->LightSourceGroupBox->Location = System::Drawing::Point(7, 166);
	this->LightSourceGroupBox->Name = S"LightSourceGroupBox";
	this->LightSourceGroupBox->Size = System::Drawing::Size(252, 73);
	this->LightSourceGroupBox->TabIndex = 113;
	this->LightSourceGroupBox->TabStop = false;
	this->LightSourceGroupBox->Text = S"Light Source";
	// 
	// LambdaMinNumeric
	// 
	this->LambdaMinNumeric->Location = System::Drawing::Point(150, 18);
	System::Int32 __mcTemp__4[] = new System::Int32[4];
	__mcTemp__4[0] = 100000;
	__mcTemp__4[1] = 0;
	__mcTemp__4[2] = 0;
	__mcTemp__4[3] = 0;
	this->LambdaMinNumeric->Maximum = System::Decimal(__mcTemp__4);
	this->LambdaMinNumeric->Name = S"LambdaMinNumeric";
	this->LambdaMinNumeric->Size = System::Drawing::Size(91, 21);
	this->LambdaMinNumeric->TabIndex = 112;
	System::Int32 __mcTemp__5[] = new System::Int32[4];
	__mcTemp__5[0] = 788;
	__mcTemp__5[1] = 0;
	__mcTemp__5[2] = 0;
	__mcTemp__5[3] = 0;
	this->LambdaMinNumeric->Value = System::Decimal(__mcTemp__5);
	// 
	// LambdaMinLabel
	// 
	this->LambdaMinLabel->AutoSize = true;
	this->LambdaMinLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->LambdaMinLabel->Location = System::Drawing::Point(6, 20);
	this->LambdaMinLabel->Name = S"LambdaMinLabel";
	this->LambdaMinLabel->Size = System::Drawing::Size(123, 15);
	this->LambdaMinLabel->TabIndex = 113;
	this->LambdaMinLabel->Text = S"Lambda Minimum";
	// 
	// dLambdaLabel
	// 
	this->dLambdaLabel->AutoSize = true;
	this->dLambdaLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->dLambdaLabel->Location = System::Drawing::Point(6, 47);
	this->dLambdaLabel->Name = S"dLambdaLabel";
	this->dLambdaLabel->Size = System::Drawing::Size(67, 15);
	this->dLambdaLabel->TabIndex = 115;
	this->dLambdaLabel->Text = S"dLambda";
	// 
	// dLambdaNumeric
	// 
	this->dLambdaNumeric->DecimalPlaces = 5;
	System::Int32 __mcTemp__6[] = new System::Int32[4];
	__mcTemp__6[0] = 1;
	__mcTemp__6[1] = 0;
	__mcTemp__6[2] = 0;
	__mcTemp__6[3] = 196608;
	this->dLambdaNumeric->Increment = System::Decimal(__mcTemp__6);
	this->dLambdaNumeric->Location = System::Drawing::Point(150, 45);
	System::Int32 __mcTemp__7[] = new System::Int32[4];
	__mcTemp__7[0] = 10;
	__mcTemp__7[1] = 0;
	__mcTemp__7[2] = 0;
	__mcTemp__7[3] = 0;
	this->dLambdaNumeric->Maximum = System::Decimal(__mcTemp__7);
	this->dLambdaNumeric->Name = S"dLambdaNumeric";
	this->dLambdaNumeric->Size = System::Drawing::Size(91, 21);
	this->dLambdaNumeric->TabIndex = 114;
	System::Int32 __mcTemp__8[] = new System::Int32[4];
	__mcTemp__8[0] = 85;
	__mcTemp__8[1] = 0;
	__mcTemp__8[2] = 0;
	__mcTemp__8[3] = 196608;
	this->dLambdaNumeric->Value = System::Decimal(__mcTemp__8);
	// 
	// ImageWidthTxtBx
	// 
	this->ImageWidthTxtBx->BackColor = System::Drawing::SystemColors::ButtonHighlight;
	this->ImageWidthTxtBx->Location = System::Drawing::Point(269, 35);
	this->ImageWidthTxtBx->Name = S"ImageWidthTxtBx";
	this->ImageWidthTxtBx->Size = System::Drawing::Size(66, 21);
	this->ImageWidthTxtBx->TabIndex = 13;
	this->ImageWidthTxtBx->Text = S"2048";
	// 
	// ImageHeightTxtBx
	// 
	this->ImageHeightTxtBx->BackColor = System::Drawing::SystemColors::ButtonHighlight;
	this->ImageHeightTxtBx->Location = System::Drawing::Point(269, 75);
	this->ImageHeightTxtBx->Name = S"ImageHeightTxtBx";
	this->ImageHeightTxtBx->Size = System::Drawing::Size(66, 21);
	this->ImageHeightTxtBx->TabIndex = 15;
	this->ImageHeightTxtBx->Text = S"512";
	// 
	// ImageWidthLabel
	// 
	this->ImageWidthLabel->Location = System::Drawing::Point(269, 17);
	this->ImageWidthLabel->Name = S"ImageWidthLabel";
	this->ImageWidthLabel->Size = System::Drawing::Size(77, 19);
	this->ImageWidthLabel->TabIndex = 14;
	this->ImageWidthLabel->Text = S"Image Width";
	// 
	// ScanningGroupBox
	// 
	this->ScanningGroupBox->BackColor = System::Drawing::Color::Transparent;
	this->ScanningGroupBox->Controls->Add(this->FlyBackPoints);
	this->ScanningGroupBox->Controls->Add(this->FlyBackPointsLabel);
	this->ScanningGroupBox->Controls->Add(this->LineRateNumeric);
	this->ScanningGroupBox->Controls->Add(this->YscanOffset_mVNumeric);
	this->ScanningGroupBox->Controls->Add(this->BScanFrameRateLabel);
	this->ScanningGroupBox->Controls->Add(this->XScanAmp_mVNumeric);
	this->ScanningGroupBox->Controls->Add(this->FrameScanLabel);
	this->ScanningGroupBox->Controls->Add(this->XAmp_mV_Label);
	this->ScanningGroupBox->Controls->Add(this->XScanOffset_mVNumeric);
	this->ScanningGroupBox->Controls->Add(this->AmplitudeLabel);
	this->ScanningGroupBox->Controls->Add(this->XOffset_mV_Label);
	this->ScanningGroupBox->Controls->Add(this->YscanAmp_mVNumeric);
	this->ScanningGroupBox->Controls->Add(this->OffsetLabel);
	this->ScanningGroupBox->Controls->Add(this->YOffset_mV_Label);
	this->ScanningGroupBox->Controls->Add(this->YAmp_mV_Label);
	this->ScanningGroupBox->Controls->Add(this->YscanOffChckBx);
	this->ScanningGroupBox->Location = System::Drawing::Point(7, 4);
	this->ScanningGroupBox->Name = S"ScanningGroupBox";
	this->ScanningGroupBox->Size = System::Drawing::Size(252, 156);
	this->ScanningGroupBox->TabIndex = 109;
	this->ScanningGroupBox->TabStop = false;
	this->ScanningGroupBox->Text = S"Scanning";
	// 
	// FlyBackPoints
	// 
	this->FlyBackPoints->Location = System::Drawing::Point(150, 100);
	System::Int32 __mcTemp__9[] = new System::Int32[4];
	__mcTemp__9[0] = 200000;
	__mcTemp__9[1] = 0;
	__mcTemp__9[2] = 0;
	__mcTemp__9[3] = 0;
	this->FlyBackPoints->Maximum = System::Decimal(__mcTemp__9);
	System::Int32 __mcTemp__10[] = new System::Int32[4];
	__mcTemp__10[0] = 1;
	__mcTemp__10[1] = 0;
	__mcTemp__10[2] = 0;
	__mcTemp__10[3] = 0;
	this->FlyBackPoints->Minimum = System::Decimal(__mcTemp__10);
	this->FlyBackPoints->Name = S"FlyBackPoints";
	this->FlyBackPoints->Size = System::Drawing::Size(91, 21);
	this->FlyBackPoints->TabIndex = 119;
	System::Int32 __mcTemp__11[] = new System::Int32[4];
	__mcTemp__11[0] = 50;
	__mcTemp__11[1] = 0;
	__mcTemp__11[2] = 0;
	__mcTemp__11[3] = 0;
	this->FlyBackPoints->Value = System::Decimal(__mcTemp__11);
	this->FlyBackPoints->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// FlyBackPointsLabel
	// 
	this->FlyBackPointsLabel->AutoSize = true;
	this->FlyBackPointsLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.5F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->FlyBackPointsLabel->Location = System::Drawing::Point(6, 104);
	this->FlyBackPointsLabel->Name = S"FlyBackPointsLabel";
	this->FlyBackPointsLabel->Size = System::Drawing::Size(100, 15);
	this->FlyBackPointsLabel->TabIndex = 45;
	this->FlyBackPointsLabel->Text = S"FlyBack Points";
	// 
	// LineRateNumeric
	// 
	this->LineRateNumeric->Location = System::Drawing::Point(150, 125);
	System::Int32 __mcTemp__12[] = new System::Int32[4];
	__mcTemp__12[0] = 200000;
	__mcTemp__12[1] = 0;
	__mcTemp__12[2] = 0;
	__mcTemp__12[3] = 0;
	this->LineRateNumeric->Maximum = System::Decimal(__mcTemp__12);
	System::Int32 __mcTemp__13[] = new System::Int32[4];
	__mcTemp__13[0] = 1;
	__mcTemp__13[1] = 0;
	__mcTemp__13[2] = 0;
	__mcTemp__13[3] = 0;
	this->LineRateNumeric->Minimum = System::Decimal(__mcTemp__13);
	this->LineRateNumeric->Name = S"LineRateNumeric";
	this->LineRateNumeric->Size = System::Drawing::Size(91, 21);
	this->LineRateNumeric->TabIndex = 31;
	System::Int32 __mcTemp__14[] = new System::Int32[4];
	__mcTemp__14[0] = 360;
	__mcTemp__14[1] = 0;
	__mcTemp__14[2] = 0;
	__mcTemp__14[3] = 0;
	this->LineRateNumeric->Value = System::Decimal(__mcTemp__14);
	this->LineRateNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// YscanOffset_mVNumeric
	// 
	System::Int32 __mcTemp__15[] = new System::Int32[4];
	__mcTemp__15[0] = 50;
	__mcTemp__15[1] = 0;
	__mcTemp__15[2] = 0;
	__mcTemp__15[3] = 0;
	this->YscanOffset_mVNumeric->Increment = System::Decimal(__mcTemp__15);
	this->YscanOffset_mVNumeric->Location = System::Drawing::Point(162, 67);
	System::Int32 __mcTemp__16[] = new System::Int32[4];
	__mcTemp__16[0] = 5000;
	__mcTemp__16[1] = 0;
	__mcTemp__16[2] = 0;
	__mcTemp__16[3] = 0;
	this->YscanOffset_mVNumeric->Maximum = System::Decimal(__mcTemp__16);
	System::Int32 __mcTemp__17[] = new System::Int32[4];
	__mcTemp__17[0] = 5000;
	__mcTemp__17[1] = 0;
	__mcTemp__17[2] = 0;
	__mcTemp__17[3] = System::Int32::MinValue;
	this->YscanOffset_mVNumeric->Minimum = System::Decimal(__mcTemp__17);
	this->YscanOffset_mVNumeric->Name = S"YscanOffset_mVNumeric";
	this->YscanOffset_mVNumeric->Size = System::Drawing::Size(56, 21);
	this->YscanOffset_mVNumeric->TabIndex = 73;
	this->YscanOffset_mVNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// BScanFrameRateLabel
	// 
	this->BScanFrameRateLabel->AutoSize = true;
	this->BScanFrameRateLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.5F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->BScanFrameRateLabel->Location = System::Drawing::Point(6, 129);
	this->BScanFrameRateLabel->Name = S"BScanFrameRateLabel";
	this->BScanFrameRateLabel->Size = System::Drawing::Size(128, 15);
	this->BScanFrameRateLabel->TabIndex = 33;
	this->BScanFrameRateLabel->Text = S"B-Scan FrameRate";
	// 
	// XScanAmp_mVNumeric
	// 
	System::Int32 __mcTemp__18[] = new System::Int32[4];
	__mcTemp__18[0] = 50;
	__mcTemp__18[1] = 0;
	__mcTemp__18[2] = 0;
	__mcTemp__18[3] = 0;
	this->XScanAmp_mVNumeric->Increment = System::Decimal(__mcTemp__18);
	this->XScanAmp_mVNumeric->Location = System::Drawing::Point(73, 36);
	System::Int32 __mcTemp__19[] = new System::Int32[4];
	__mcTemp__19[0] = 10000;
	__mcTemp__19[1] = 0;
	__mcTemp__19[2] = 0;
	__mcTemp__19[3] = 0;
	this->XScanAmp_mVNumeric->Maximum = System::Decimal(__mcTemp__19);
	this->XScanAmp_mVNumeric->Name = S"XScanAmp_mVNumeric";
	this->XScanAmp_mVNumeric->Size = System::Drawing::Size(54, 21);
	this->XScanAmp_mVNumeric->TabIndex = 48;
	System::Int32 __mcTemp__20[] = new System::Int32[4];
	__mcTemp__20[0] = 2500;
	__mcTemp__20[1] = 0;
	__mcTemp__20[2] = 0;
	__mcTemp__20[3] = 0;
	this->XScanAmp_mVNumeric->Value = System::Decimal(__mcTemp__20);
	this->XScanAmp_mVNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// FrameScanLabel
	// 
	this->FrameScanLabel->AutoSize = true;
	this->FrameScanLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.25F, (System::Drawing::FontStyle)(System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Underline), 
		System::Drawing::GraphicsUnit::Point, (System::Byte)0));
	this->FrameScanLabel->Location = System::Drawing::Point(69, 16);
	this->FrameScanLabel->Name = S"FrameScanLabel";
	this->FrameScanLabel->Size = System::Drawing::Size(74, 13);
	this->FrameScanLabel->TabIndex = 49;
	this->FrameScanLabel->Text = S"Frame Scan";
	// 
	// XAmp_mV_Label
	// 
	this->XAmp_mV_Label->AutoSize = true;
	this->XAmp_mV_Label->Location = System::Drawing::Point(126, 39);
	this->XAmp_mV_Label->Name = S"XAmp_mV_Label";
	this->XAmp_mV_Label->Size = System::Drawing::Size(25, 15);
	this->XAmp_mV_Label->TabIndex = 50;
	this->XAmp_mV_Label->Text = S"mV";
	// 
	// XScanOffset_mVNumeric
	// 
	System::Int32 __mcTemp__21[] = new System::Int32[4];
	__mcTemp__21[0] = 50;
	__mcTemp__21[1] = 0;
	__mcTemp__21[2] = 0;
	__mcTemp__21[3] = 0;
	this->XScanOffset_mVNumeric->Increment = System::Decimal(__mcTemp__21);
	this->XScanOffset_mVNumeric->Location = System::Drawing::Point(73, 67);
	System::Int32 __mcTemp__22[] = new System::Int32[4];
	__mcTemp__22[0] = 5000;
	__mcTemp__22[1] = 0;
	__mcTemp__22[2] = 0;
	__mcTemp__22[3] = 0;
	this->XScanOffset_mVNumeric->Maximum = System::Decimal(__mcTemp__22);
	System::Int32 __mcTemp__23[] = new System::Int32[4];
	__mcTemp__23[0] = 5000;
	__mcTemp__23[1] = 0;
	__mcTemp__23[2] = 0;
	__mcTemp__23[3] = System::Int32::MinValue;
	this->XScanOffset_mVNumeric->Minimum = System::Decimal(__mcTemp__23);
	this->XScanOffset_mVNumeric->Name = S"XScanOffset_mVNumeric";
	this->XScanOffset_mVNumeric->Size = System::Drawing::Size(54, 21);
	this->XScanOffset_mVNumeric->TabIndex = 51;
	this->XScanOffset_mVNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// AmplitudeLabel
	// 
	this->AmplitudeLabel->AutoSize = true;
	this->AmplitudeLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->AmplitudeLabel->Location = System::Drawing::Point(5, 39);
	this->AmplitudeLabel->Name = S"AmplitudeLabel";
	this->AmplitudeLabel->Size = System::Drawing::Size(62, 13);
	this->AmplitudeLabel->TabIndex = 52;
	this->AmplitudeLabel->Text = S"Amplitude";
	// 
	// XOffset_mV_Label
	// 
	this->XOffset_mV_Label->AutoSize = true;
	this->XOffset_mV_Label->Location = System::Drawing::Point(126, 70);
	this->XOffset_mV_Label->Name = S"XOffset_mV_Label";
	this->XOffset_mV_Label->Size = System::Drawing::Size(25, 15);
	this->XOffset_mV_Label->TabIndex = 53;
	this->XOffset_mV_Label->Text = S"mV";
	// 
	// YscanAmp_mVNumeric
	// 
	this->YscanAmp_mVNumeric->Location = System::Drawing::Point(162, 36);
	System::Int32 __mcTemp__24[] = new System::Int32[4];
	__mcTemp__24[0] = 10000;
	__mcTemp__24[1] = 0;
	__mcTemp__24[2] = 0;
	__mcTemp__24[3] = 0;
	this->YscanAmp_mVNumeric->Maximum = System::Decimal(__mcTemp__24);
	this->YscanAmp_mVNumeric->Name = S"YscanAmp_mVNumeric";
	this->YscanAmp_mVNumeric->Size = System::Drawing::Size(56, 21);
	this->YscanAmp_mVNumeric->TabIndex = 72;
	System::Int32 __mcTemp__25[] = new System::Int32[4];
	__mcTemp__25[0] = 2500;
	__mcTemp__25[1] = 0;
	__mcTemp__25[2] = 0;
	__mcTemp__25[3] = 0;
	this->YscanAmp_mVNumeric->Value = System::Decimal(__mcTemp__25);
	this->YscanAmp_mVNumeric->ValueChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// OffsetLabel
	// 
	this->OffsetLabel->AutoSize = true;
	this->OffsetLabel->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->OffsetLabel->Location = System::Drawing::Point(5, 72);
	this->OffsetLabel->Name = S"OffsetLabel";
	this->OffsetLabel->Size = System::Drawing::Size(41, 13);
	this->OffsetLabel->TabIndex = 75;
	this->OffsetLabel->Text = S"Offset";
	// 
	// YOffset_mV_Label
	// 
	this->YOffset_mV_Label->AutoSize = true;
	this->YOffset_mV_Label->Location = System::Drawing::Point(216, 68);
	this->YOffset_mV_Label->Name = S"YOffset_mV_Label";
	this->YOffset_mV_Label->Size = System::Drawing::Size(25, 15);
	this->YOffset_mV_Label->TabIndex = 80;
	this->YOffset_mV_Label->Text = S"mV";
	// 
	// YAmp_mV_Label
	// 
	this->YAmp_mV_Label->AutoSize = true;
	this->YAmp_mV_Label->Location = System::Drawing::Point(216, 39);
	this->YAmp_mV_Label->Name = S"YAmp_mV_Label";
	this->YAmp_mV_Label->Size = System::Drawing::Size(25, 15);
	this->YAmp_mV_Label->TabIndex = 81;
	this->YAmp_mV_Label->Text = S"mV";
	// 
	// YscanOffChckBx
	// 
	this->YscanOffChckBx->AutoSize = true;
	this->YscanOffChckBx->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->YscanOffChckBx->Location = System::Drawing::Point(149, 15);
	this->YscanOffChckBx->Name = S"YscanOffChckBx";
	this->YscanOffChckBx->Size = System::Drawing::Size(100, 17);
	this->YscanOffChckBx->TabIndex = 71;
	this->YscanOffChckBx->Text = S"Volume Scan";
	this->YscanOffChckBx->UseVisualStyleBackColor = true;
	this->YscanOffChckBx->CheckedChanged += new System::EventHandler(this, &Form1::ScanValChanged);
	// 
	// ImageHeightLabel
	// 
	this->ImageHeightLabel->Location = System::Drawing::Point(269, 57);
	this->ImageHeightLabel->Name = S"ImageHeightLabel";
	this->ImageHeightLabel->Size = System::Drawing::Size(84, 16);
	this->ImageHeightLabel->TabIndex = 16;
	this->ImageHeightLabel->Text = S"Image Height";
	// 
	// allTabs
	// 
	this->allTabs->Controls->Add(this->OCTMainContTab);
	this->allTabs->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->allTabs->Location = System::Drawing::Point(20, 59);
	this->allTabs->Name = S"allTabs";
	this->allTabs->SelectedIndex = 0;
	this->allTabs->Size = System::Drawing::Size(366, 270);
	this->allTabs->TabIndex = 110;
	// 
	// hotKeysGroupBox
	// 
	this->hotKeysGroupBox->Controls->Add(this->hotKeyMenu);
	this->hotKeysGroupBox->Location = System::Drawing::Point(397, 12);
	this->hotKeysGroupBox->Name = S"hotKeysGroupBox";
	this->hotKeysGroupBox->Size = System::Drawing::Size(336, 424);
	this->hotKeysGroupBox->TabIndex = 112;
	this->hotKeysGroupBox->TabStop = false;
	this->hotKeysGroupBox->Text = S"Hotkeys";
	// 
	// hotKeyMenu
	// 
	this->hotKeyMenu->Enabled = false;
	this->hotKeyMenu->FormattingEnabled = true;
	this->hotKeyMenu->ItemHeight = 16;
	System::Object* __mcTemp__26[] = new System::Object*[19];
	__mcTemp__26[0] = S"*Ensure CAPS LOCK is OFF";
	__mcTemp__26[1] = S"";
	__mcTemp__26[2] = S"d --- Acquire new DC";
	__mcTemp__26[3] = S"a --- Averaging ON/OFF";
	__mcTemp__26[4] = S"b --- Bilateral Filter ON/OFF";
	__mcTemp__26[5] = S"f / g --- Switch Interpolation Method";
	__mcTemp__26[6] = S"n / m--- decr/incr fundus brightness";
	__mcTemp__26[7] = S"r --- reset all display";
	__mcTemp__26[8] = S"- / = --- decr/incr range min value ";
	__mcTemp__26[9] = S"[ / ] --- decr/incr range max value ";
	__mcTemp__26[10] = S"; / \' --- decr/incr voxel threshold";
	__mcTemp__26[11] = S", / . --- decr/incr dispersion value";
	__mcTemp__26[12] = S"home --- switch to crop/downsize mode";
	__mcTemp__26[13] = S"end --- (Crop) min/max crop size";
	__mcTemp__26[14] = S"left --- (Crop) decr vol size OR (DS) downsize";
	__mcTemp__26[15] = S"right --- (Crop) incr vol size OR (DS) upsize";
	__mcTemp__26[16] = S"down --- (Crop) shift volume down";
	__mcTemp__26[17] = S"up --- (Crop) shift volume up";
	__mcTemp__26[18] = S"esc--- Exit program";
	this->hotKeyMenu->Items->AddRange(__mcTemp__26);
	this->hotKeyMenu->Location = System::Drawing::Point(16, 26);
	this->hotKeyMenu->Name = S"hotKeyMenu";
	this->hotKeyMenu->Size = System::Drawing::Size(301, 372);
	this->hotKeyMenu->TabIndex = 0;
	// 
	// Form1
	// 
	this->AutoScaleBaseSize = System::Drawing::Size(6, 15);
	this->BackColor = System::Drawing::SystemColors::Control;
	this->ClientSize = System::Drawing::Size(745, 444);
	this->Controls->Add(this->hotKeysGroupBox);
	this->Controls->Add(this->SavingGroupBox);
	this->Controls->Add(this->NewFileCheckBox);
	this->Controls->Add(this->acquireSelectionComboBox);
	this->Controls->Add(this->StartThreadButton);
	this->Controls->Add(this->StopButton);
	this->Controls->Add(this->StatusTextBox);
	this->Controls->Add(this->allTabs);
	this->Font = (new System::Drawing::Font(S"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
		(System::Byte)0));
	this->Name = S"Form1";
	this->Text = S"SFU BORG OCTViewer";
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YFrameNumNumeric))->EndInit();
	this->SavingGroupBox->ResumeLayout(false);
	this->SavingGroupBox->PerformLayout();
	this->OCTMainContTab->ResumeLayout(false);
	this->OCTMainContTab->PerformLayout();
	this->LightSourceGroupBox->ResumeLayout(false);
	this->LightSourceGroupBox->PerformLayout();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->LambdaMinNumeric))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->dLambdaNumeric))->EndInit();
	this->ScanningGroupBox->ResumeLayout(false);
	this->ScanningGroupBox->PerformLayout();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->FlyBackPoints))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->LineRateNumeric))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YscanOffset_mVNumeric))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->XScanAmp_mVNumeric))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->XScanOffset_mVNumeric))->EndInit();
	(__try_cast<System::ComponentModel::ISupportInitialize*  >(this->YscanAmp_mVNumeric))->EndInit();
	this->allTabs->ResumeLayout(false);
	this->hotKeysGroupBox->ResumeLayout(false);
	this->ResumeLayout(false);
	this->PerformLayout();

}

};
}