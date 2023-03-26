# beGAN
 Batch-effect Elimination Generative Adversarial Network
 
 **Eyecatching and Impressive images** 
 
 **Cite our paper**
 
 **Brief intro on the use of beGAN.**


## Setup

### Prerequisites
- Window
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- Microsoft Visual Studio
- Anaconda3

### Getting Started
- Open Anaconda Prompt and create a new virtual enviroment named as BeGAN_GPU
```
conda create --name BeGAN_GPU
```
- Activate the enviroment and install `python`
```
conda activate BeGAN_GPU
conda install python=3.8
```
- Install `cudnn`, `pip` and `tensorflow-gpu`
```
conda install -c anaconda cudnn
conda install pip
pip install tensorflow-gpu==2.4
```
- Install all the packages required for running beGAN codes (`Pillow`, `matplotlib`, `scipy`, `opencv`, `scikit-learn`, `pandas`, `imageio` and `mat73`)
```
conda install Pillow=9.2
conda install -c conda-forge matplotlib==3.5.2
conda install -c anaconda scipy 
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c anaconda pandas
conda install -c conda-forge imageio
pip install mat73
```
The BeGAN_GPU virtual enviroment is ready for training and testing.

## 7 lung Cancer Cell Lines Dataset
The 7 lung cancer cell lines dataset is uploaded in this repository. It is used as an demonstration on batch removal and image contrast conversion of the beGAN model. 

**7 Lung Cancer Cell Lines BF and QPI Images**

There are in total of 7 types of lung cancer cells (i.e. H69, H358, H520, H526, H1975, H2170 and HCC827). All the data were collected on 7 days using [multi-ATOM setup](https://doi.org/10.1002/jbio.201800479), giving 3 batches per cell lines. Both single-cell brightfield and quantitative phase images (QPI) were collected.

For training and testing the beGAN model, the data were separated into "Train", "Valid" and "Test" set, each of them containing 1000, 1000 and 7000 cells respectively. Data was uploaded in `.mat` format with brightfield images in `_BF.mat` and QPI in `_QPI.mat`. The images are stored in format of `ImageHeight * ImageWidth * NoOfCells` with a field of view of 45μm.

## Training the BeGAN Model


## Load and Test the BeGAN Model

