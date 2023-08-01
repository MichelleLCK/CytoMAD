## 7 lung Cancer Cell Lines Dataset
The 7 lung cancer cell lines dataset is uploaded in this repository. It is used as an demonstration on batch removal and image contrast conversion of the CytoMAD model. 

![LungCancerCellLinesResult](https://github.com/MichelleLCK/CytoMAD/assets/120153122/cfcd0932-c53e-4fa5-abce-a58a1da78cb0)

There are in total of 7 types of lung cancer cells (i.e. H69, H358, H520, H526, H1975, H2170 and HCC827). All the data were collected on 7 days using [multi-ATOM setup](https://doi.org/10.1002/jbio.201800479), giving 3 batches per cell lines. Both single-cell brightfield and quantitative phase images (QPI) were collected.

For training and testing the CytoMAD model, the data were separated into "Train", "Valid" and "Test" set. They are subsampled and contain 500 cells respectively as a demonstration in this repository (Folder `Dataset`). Data was uploaded in `.mat` format with brightfield images in `_BF.mat` and QPI in `_QPI.mat`. The images are stored in format of `ImageHeight * ImageWidth * NoOfCells` with a field of view of 45μm. 

<sub>Full dataset will be released upon request.</sub>
