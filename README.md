# Heart-Disease-Prediction-Using-ANN-2

The dataset used for this project describes diagnosing of cardiac Single Proton Emission Computed Tomography (SPECT) images. 
Each of the patients is classified into two categories: normal(1) and abnormal(0). 

The parallizing is done using OpenMP API.

**SerialTraining.c** contains the serial code which takes approximately 83 seconds to run 100000 epochs.

**ParallelizedTraining.c** contains the parallelized code that takes approximately 58 seconds to run 100000 epochs which shows the significant speed-up when comared to the corresponding serial code.
 
