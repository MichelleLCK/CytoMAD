# BeGAN
 Batch-effect Elimination Generative Adversarial Network
 
 **Eyecatching and Impressive images** 
 
 **Cite our paper**
 
 **Brief intro on the use of beGAN.**
The beGAN model is a generative deep learning model that integrates conditional GAN of [Pix2Pix](https://doi.org/10.1109/CVPR.2017.632) architect with the classification networks, which altogether enables robust cell images conversion among different image contrast and achieves batch information removal on the image level.

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
Overall, the training of beGAN model consist of 2 main parts. It utilizes the conditional GAN of Pix2Pix model as backbone with the addition of classifier networks to guide the batch-effect-removal.

### Pre-training of the Pix2Pix Backbone
The basic framework of conditional GAN takes in cell images of particular imaging contrasts (e.g. brightfield) as model input. 

The images would then be directed to the `generator` and undergo multiple layers of 2-dimensional (2D) convolutional layers, normalization layers and mathematical functions, eventually be condensed into a 1-dimensional (1D) array at the bottleneck. Such array only describes the important features extracted from the cell images. The output images of the model, which are the cell images of different imaging contrast (e.g. QPI), are reconstructed based on these concise 1D information through deconvolutional layers and mathematical equations. 
```
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g
 
# define the generator model
def define_generator(image_shape=(256, 256, 1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model    
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e6, 512)
	d2 = decoder_block(d1, e5, 512)
	d3 = decoder_block(d2, e4, 512, dropout=False)
	d4 = decoder_block(d3, e3, 256, dropout=False)
	d5 = decoder_block(d4, e2, 128, dropout=False)
	d6 = decoder_block(d5, e1, 64, dropout=False)

	# output
	d7 = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6)
	d7 = BatchNormalization()(d7, training=True)
	d7 = Concatenate()([d7, in_image])
	d7 = Activation('relu')(d7)
	d8 = Conv2DTranspose(16, (4,4), padding='same', kernel_initializer=init)(d7)
	d8 = BatchNormalization()(d8, training=True)
	d8 = Activation('relu')(d8)
	d9 = Conv2DTranspose(8, (4,4), padding='same', kernel_initializer=init)(d8)
	d9 = BatchNormalization()(d9, training=True)
	d9 = Activation('relu')(d9)
	d10 = Conv2DTranspose(4, (4,4), padding='same', kernel_initializer=init)(d9)
	d10 = BatchNormalization()(d10, training=True)
	d10 = Activation('relu')(d10)
	g = Conv2DTranspose(1, (4,4), padding='same', kernel_initializer=init)(d10)

	out_image = Activation('linear')(g)
	# define model
	model = Model(in_image, [out_image, b])
	model.summary()
	return model
```

The `discriminator` model is a classifier which classify between the target images and the generator's output images. It is used to guide and tune to the generator for predicting accurate target images. With these, the conditional GAN converts cell images from one imaging contrast to another. 
```
# define the discriminator model
def define_discriminator(input_shape, output_shape, Lrate = 0.000001):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=input_shape)
	# target image input
	in_target_image = Input(shape=output_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
    
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=Lrate, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	model.summary()
	return model
 
 # define the pretraining beGAN model
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
    
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	[gen_out, _] = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model
```

### Classifiers-guided Batch Effect Removal
BeGAN differs itself from the ordinary conditional GAN by the additional classification networks. To remove batch-to-batch variations while preserving biological differences, the classification networks takes a key role here through 2 types of classifiers, the batch classifiers and cell type classifiers. 

At the bottleneck region, the batch classifiers aim in eliminating batch-related information with the cell type classifier conserving the cellular variations in the 1D features, with the basic framework of neural network.
```
# define the neural-network-based classifier model (Cell Type Classifier)
def define_classifier(NoOfClass, input_shape = (512,), Lrate = 0.0001):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	features_gen = Input(shape=input_shape)
    
	l1 = Dense(250, activation='relu', kernel_initializer=init)(features_gen)
	l2 = Dense(100, activation='relu', kernel_initializer=init)(l1)
	l3 = Dense(50, activation='relu', kernel_initializer=init)(l2)
	l4 = Dense(25, activation='relu', kernel_initializer=init)(l3)
	out = Dense(NoOfClass, activation='softmax', kernel_initializer=init)(l4)
    
	# define model
	model = Model(features_gen, out)
	# compile model
	opt = Adam(lr=Lrate, beta_1=0.5)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model

# define the neural-network-based classifier model (Batch Classifier)
def define_classifier_dropout(NoOfClass, input_shape = (100,), Lrate = 0.0001):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	features_gen = Input(shape=input_shape)
	l1 = Dense(75, activation='relu', kernel_initializer=init)(features_gen)
	l2 = Dense(50, activation='relu', kernel_initializer=init)(l1)
	l3 = Dense(25, activation='relu', kernel_initializer=init)(l2)
	out = Dense(NoOfClass, activation='softmax', kernel_initializer=init)(l3)
    
	# define model
	model = Model(features_gen, out)
	# compile model
	opt = Adam(lr=Lrate, beta_1=0.5)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model
```

The batch and cell type classifiers that based on convolutional neural networks also present by the end of conditional GAN for guiding the reconstruction of batch-free cell images. 
```
# define the convolutional-neural-network-based classifier model
def define_CNN(input_shape, NoOfClass, Lrate = 0.001):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=input_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = MaxPooling2D((2,2), padding='same')(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = MaxPooling2D((2,2), padding='same')(d)
	# C512
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = MaxPooling2D((2,2), padding='same')(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = MaxPooling2D((2,2), padding='same')(d)
	d = Flatten()(d)
	d = Dense(128, activation='relu', kernel_initializer=init)(d)
	# patch output
	patch_out = Dense(NoOfClass, activation='softmax', kernel_initializer=init)(d)
    
	# define model
	model = Model(in_src_image, patch_out)
	# compile model
	opt = Adam(lr=Lrate, beta_1=0.5)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model
```

Overall, the classification networks forms a feedback system with the basic conditional GAN framework for disentangling the batch information from the biological variations of interest, and eventually, achieving batch-free property at both the concise 1D biophysical phenotyping and the endmost reconstructed images.
```
# define the beGAN model
def define_mapping_gan(g_model, d_model, cell_model, cell_model_CNN, batch_model_CNN, batch_model1, batch_model2, batch_model3, batch_model4, RandomFeaturesList1, RandomFeaturesList2, RandomFeaturesList3, RandomFeaturesList4, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	batch_model1.trainable = False
	batch_model2.trainable = False
	batch_model3.trainable = False
	batch_model4.trainable = False
	cell_model.trainable = False
	batch_model_CNN.trainable = False
	cell_model_CNN.trainable = False
    
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	[gen_out, gen_features] = g_model(in_src)
	features_gen = tf.squeeze(gen_features, [1, 2])
	dis_out = d_model([in_src, gen_out])
	batch_CNN_out = batch_model_CNN(gen_out)
	cell_CNN_out = cell_model_CNN(gen_out)
	batch_out1 = batch_model1(tf.gather(features_gen, RandomFeaturesList1, axis = 1))
	batch_out2 = batch_model2(tf.gather(features_gen, RandomFeaturesList2, axis = 1))
	batch_out3 = batch_model3(tf.gather(features_gen, RandomFeaturesList3, axis = 1))
	batch_out4 = batch_model4(tf.gather(features_gen, RandomFeaturesList4, axis = 1))
	cell_out = cell_model(features_gen)
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, cell_CNN_out, cell_out, batch_CNN_out, batch_out1, batch_out2, batch_out3, batch_out4, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100, 100, 100, 100, 100, 100, 100, 300])
	return model
```

The predicted images will be saved in folder `./Figures` and the trained model will be saved in folder `./ModelParameters`.
The entire code is available in [BeGAN_7LungCancer_Train.py](https://github.com/MichelleLCK/beGAN/blob/dc77129aac886a817873e2c9606a9be2fb6533da/BeGAN_7LungCancer_Train.py).

## Load and Test the BeGAN Model
Select and load the trained BeGAN model for generating batch-free images and features.
```
# load saved beGAN model
modelpath = SavePath+'\\ModelParameters'
modelfolder = 'XXXXXXXX_XXXXXX_BeGAN_With_Batch_Removal'
modelname = 'model_XXXXXX.h5'
g_model = load_model(modelpath + '\\' + modelfolder + '\\' + modelname)
```

The predicted output will be saved in folder `./TestData`.
The entire code is available in [BeGAN_7LungCancer_Test.py](https://github.com/MichelleLCK/beGAN/blob/dc77129aac886a817873e2c9606a9be2fb6533da/BeGAN_7LungCancer_Test.py).
