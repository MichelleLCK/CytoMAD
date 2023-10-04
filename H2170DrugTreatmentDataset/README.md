## Drug Treatment Response of Lung Cancer Cell H2170 Dataset
The drug treatment response of H2170 dataset is uploaded in this repository. It serves as another demonstration on batch removal and image contrast conversion of the CytoMAD model. 

![LungCancerDrugResult](https://github.com/MichelleLCK/CytoMAD/assets/120153122/cc0db26b-473b-4bc7-ab60-c7533ee89b1d)


In this experiment, H2170 were treated with 3 drugs of different mechanism of action (MoA) (i.e. Docetaxel, Afatinib and Gemcitabine), each with 5 concentration levels and a negative control with dimethyl sulfoxide (DMSO) for 24 hours as listed below. They were imaged using [multi-ATOM setup](https://doi.org/10.1002/jbio.201800479) for single-cell BF and QPI images on 6 days, forming 2 batches with ~100,000 cells per drug. Basically, this dataset consists of 2 batches of data, with each batch containing 3 different drug treatments and each treatment comprising 6 different concentration conditions. This results in 18 unique drug treatment conditions in each batch.


![LungCancerDrugTable](https://github.com/MichelleLCK/CytoMAD/assets/120153122/d63df4d4-eafc-47f3-8759-73459f2c4229)


For training and testing the CytoMAD model, the data were separated into "Train", "Valid" and "Test" set. They are subsampled and contain 400 cells respectively as a demonstration in this repository (Folder `Dataset`). Data was uploaded in `.mat` format with brightfield images in `_BF.mat` and QPI in `_QPI.mat`. The images are stored in the format of `ImageHeight * ImageWidth * NoOfCells` with a field of view of 45μm. 

<sub>Full dataset will be released upon request.</sub>

The entire training code and test code are available in [CytoMAD_LungCancerDrug_Train.py](https://github.com/MichelleLCK/CytoMAD/blob/main/CytoMAD_LungCancerDrug_Train.py) and [CytoMAD_LungCancerDrug_Test.py](https://github.com/MichelleLCK/CytoMAD/blob/main/CytoMAD_LungCancerDrug_Test.py).
