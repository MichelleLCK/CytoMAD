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
- Install all the packages required for running beGAN codes (`Pillow`, `matplotlib`, `scipy`, `mat73`, `opencv`, `scikit-learn`, `pandas` and `imageio`)
```
conda install Pillow=9.2
conda install -c conda-forge matplotlib==3.5.2
conda install -c anaconda scipy 
pip install mat73
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c anaconda pandas
conda install -c conda-forge imageio
```
The BeGAN_GPU virtual enviroment is ready for training and test.
