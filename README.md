# beGAN
 Batch-effect Elimination Generative Adversarial Network
 
 Impressive images 
 
 Cite our paper
 
 Brief intro on the use of beGAN.


## Setup

### Prerequisites
- Window
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- Anaconda3

### Getting Started
- Open Anaconda Prompt and create a new virtual enviroment named as BeGAN_GPU
```
conda create --name BeGAN_GPU
```
- Activate the enviroment and install python 3.8
```
conda activate BeGAN_GPU
conda install python=3.8
```

- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph` and `display`
```bash
luarocks install nngraph
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
