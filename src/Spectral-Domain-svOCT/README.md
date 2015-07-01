# Spectral-Domain-svOCT
Source code for the GPU based spectral domain svOCT, file-read based. 


Simon Fraser University, Biomedical Optics Research Group (BORG) Webpage: http://borg.ensc.sfu.ca/

GPU-svOCT Research Website: http://borg.ensc.sfu.ca/research/svoct-gpu-code.html
GPU-OCT Research Website: http://borg.ensc.sfu.ca/research/fdoct-gpu-code.html
NEWS: Google code has DEPRECATED the 'downloads' functionality. All currently existing downloads will remain in the downloads section. Future updates will be uploaded to the 'Source' section, or at the following link:

https://code.google.com/p/fdoct-gpu-code/source/browse/#svn%2Ftrunk

Open Source Code Status:

Completed:

svOCT - FileRead-Based Swept Source (Jan 20, 2014)
svOCT - FileRead-Based Spectral Domain (Jan 20, 2014)
FileRead (Feb 03, 2013)
Basler (Feb 03, 2013)
AlazarTech (Feb 01, 2014 - Refer to 'Source Tab')
Pending:

Dalsa
Research Paper Abstract:

In this paper, we describe how to highly optimize a CUDA based platform to perform real-time processing of optical coherence tomography interferometric data and 3D volumetric rendering using commercially-available cost-effective graphic processing units (GPUs). The maximum complete attainable axial scan processing rate (including memory transfer and displaying Bscan frame) was 2.24 megahertz for 16 bits pixel depth and 2048 FFT size; the maximum 3D volumetric rendering (including B-scan, en face view display, and 3D rendering) rate was ~23 volumes/second (volume size:1024x256x200). To the best of our knowledge, this is the fastest processing rate reported to date with a single-chip GPU and the first implementation of realtime video-rate volumetric OCT processing and rendering that is capable of matching the acquisition rates of ultrahigh-speed OCT.

Reference:

Jian Y, Wong K, Sarunic MV; Graphics processing unit accelerated optical coherence tomography processing at megahertz axial scan rate and high resolution video rate volumetric rendering. J. Biomed. Opt. 0001;18(2):026002-026002. doi:10.1117/1.JBO.18.2.026002.
