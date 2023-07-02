#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat March 25 13:52:03 2023

@author: Michelle C.K. lo
"""
# example of CytoMAD with image contrast conversion from single-cell brightfield to QPI, together with batch removal
from numpy import zeros
from numpy import ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow import one_hot
from matplotlib import pyplot
import cv2
import os
import numpy as np
import random
import time
import mat73
import tensorflow as tf
from sklearn.metrics import accuracy_score

 
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

# define the pretraining CytoMAD model
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

# define the CytoMAD model
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

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB, cell_label, cell_onehot, _, batch_label, batch_onehot, _ = dataset
	# choose random instances
	ix = random.sample(range(trainA.shape[0]), n_samples)
	# retrieve selected images
	X1, X2, C1, C_onehot, B1, B_onehot = trainA[ix,:,:,:], trainB[ix,:,:,:], cell_label[ix,:], cell_onehot[ix,:], batch_label[ix,:], batch_onehot[ix,:]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2, C1, C_onehot, B1, B_onehot], y

# select all samples, returns images and target
def generate_real_samples_all(dataset, patch_shape):
	# unpack dataset
	trainA, trainB, cell_label, cell_onehot, _, batch_label, batch_onehot, _ = dataset
	# retrieve all images
	X1, X2, C1, C_onehot, B1, B_onehot = trainA, trainB, cell_label, cell_onehot, batch_label, batch_onehot
	# generate 'real' class labels (1)
	n_samples = trainA.shape[0]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2, C1, C_onehot, B1, B_onehot], y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	[X, features] = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, features, y

# generate fake batch label for guiding batch removal
def generate_fake_batchlabel(batch_onehot):
	# create 'fake' batch labels
	y = zeros((batch_onehot.shape))
	y[:,0] = 1
	return y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, Path_Figures, Path_MP, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB, _, cell_onehot, _, batch_onehot], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, features_gen, _ = generate_fake_samples(g_model, X_realA, 1)
	features_gen = np.squeeze(features_gen, axis=2)
	features_gen = np.squeeze(features_gen, axis=1)
    
	NoOfNorm = X_realA.shape[3]
	NoOfPlot = NoOfNorm +2
    # plot real source images
	for i in range(NoOfNorm):
		for ii in range(n_samples):
			pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*i + ii)
			pyplot.axis('off')
			Image = X_realA[ii]
			Image = Image[:,:,i]
			pyplot.imshow(Image, cmap="gray")
	# plot predicted target image
	for i in range(n_samples):
		pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*(NoOfPlot-2) + i)
		pyplot.axis('off')
		Image = X_fakeB[i]
		Image = Image[:,:,0]
		pyplot.imshow(Image)
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*(NoOfPlot-1) + i)
		pyplot.axis('off')
		Image = X_realB[i]
		Image = Image[:,:,0]
		pyplot.imshow(Image)
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(Path_Figures +'\\'+filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(Path_MP + '\\'+filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
    
# generate CytoMAD samples and save as a plot and save the models
def summarize_performance_all(step, d_model, g_model, cell_model_CNN, cell_model, batch_model_CNN, batch_model1, batch_model2, batch_model3, batch_model4, RandomFeaturesList1, RandomFeaturesList2, RandomFeaturesList3, RandomFeaturesList4, dataset, Path_Figures_CytoMAD, Path_MP_CytoMAD, n_samples=3):

	[X_realA, X_realB, _, _, _, _], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, features_gen, _ = generate_fake_samples(g_model, X_realA, 1)
	features_gen = np.squeeze(features_gen, axis=2)
	features_gen = np.squeeze(features_gen, axis=1)

	# plot real source images
	NoOfNorm = X_realA.shape[3]
	NoOfPlot = NoOfNorm +2
	for i in range(NoOfNorm):
		for ii in range(n_samples):
			pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*i + ii)
			pyplot.axis('off')
			Image = X_realA[ii]
			Image = Image[:,:,i]
			pyplot.imshow(Image, cmap="gray")

	# plot predicted target image
	for i in range(n_samples):
		pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*(NoOfPlot-2) + i)
		pyplot.axis('off')
		Image = X_fakeB[i]
		Image = Image[:,:,0]
		pyplot.imshow(Image)
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(NoOfPlot, n_samples, 1 + n_samples*(NoOfPlot-1) + i)
		pyplot.axis('off')
		Image = X_realB[i]
		Image = Image[:,:,0]
		pyplot.imshow(Image)
	# save plot to file
	filename1 = 'plot_all_%06d.png' % (step+1)
	pyplot.savefig(Path_Figures_CytoMAD +'\\'+filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'CytoMADmodel_%06d.h5' % (step+1)
	g_model.save(Path_MP_CytoMAD +'\\'+filename2)
	# save the discriminator model
	filename3 = 'dmodel_all_%06d.h5' % (step+1)
	d_model.save(Path_MP_CytoMAD +'\\'+filename3)

	# summarize performance
	[X_realA, _, _, cell_onehot, _, batch_onehot], _ = generate_real_samples(dataset, dataset[0].shape[0], 1)
	X_fakeB, features_gen, _ = generate_fake_samples(g_model, X_realA, 1)
	features_gen = np.squeeze(features_gen, axis=2)
	features_gen = np.squeeze(features_gen, axis=1)
	print('Valid features_gen:', features_gen.shape)
	fake_batch_onehot = generate_fake_batchlabel(batch_onehot)
	batch_predict1 = batch_model1.predict(features_gen[:,RandomFeaturesList1])
	batch_predict2 = batch_model2.predict(features_gen[:,RandomFeaturesList2])
	batch_predict3 = batch_model3.predict(features_gen[:,RandomFeaturesList3])
	batch_predict4 = batch_model4.predict(features_gen[:,RandomFeaturesList4])
	cell_predict = cell_model.predict(features_gen)
	batch_CNN_predict = batch_model_CNN.predict(X_fakeB)
	cell_CNN_predict = cell_model_CNN.predict(X_fakeB)
	fake_batch_label = np.argmax(fake_batch_onehot, axis=1)
	cell_label = np.argmax(cell_onehot, axis=1)
	batch_predict1 = np.argmax(batch_predict1, axis=1)
	batch_predict2 = np.argmax(batch_predict2, axis=1)
	batch_predict3 = np.argmax(batch_predict3, axis=1)
	batch_predict4 = np.argmax(batch_predict4, axis=1)
	cell_predict = np.argmax(cell_predict, axis=1)
	batch_CNN_predict = np.argmax(batch_CNN_predict, axis=1)
	cell_CNN_predict = np.argmax(cell_CNN_predict, axis=1)
	from sklearn.metrics import accuracy_score
	b_accuracy1 = accuracy_score(fake_batch_label, batch_predict1)
	b_accuracy2 = accuracy_score(fake_batch_label, batch_predict2)
	b_accuracy3 = accuracy_score(fake_batch_label, batch_predict3)
	b_accuracy4 = accuracy_score(fake_batch_label, batch_predict4)
	c_accuracy = accuracy_score(cell_label, cell_predict)
	b_CNN_accuracy = accuracy_score(fake_batch_label, batch_CNN_predict)
	c_CNN_accuracy = accuracy_score(cell_label, cell_CNN_predict)
	print('Batch accuracy: 1. ', b_accuracy1, ', 2. ', b_accuracy2, ', 3. ', b_accuracy3, ', 4. ', b_accuracy4)
	print('Batch CNN accuracy:', b_CNN_accuracy)
	print('Cell accuracy:', c_accuracy)
	print('Cell CNN accuracy:', c_CNN_accuracy)

	filename4 = 'bmodel1_all_%06d_%.3f.h5' % (step+1, b_accuracy1)
	batch_model1.save(Path_MP_CytoMAD +'\\'+filename4)

	filename5 = 'bmodel2_all_%06d_%.3f.h5' % (step+1, b_accuracy2)
	batch_model2.save(Path_MP_CytoMAD +'\\'+filename5)

	filename6 = 'bmodel3_all_%06d_%.3f.h5' % (step+1, b_accuracy3)
	batch_model3.save(Path_MP_CytoMAD +'\\'+filename6)

	filename7 = 'bmodel4_all_%06d_%.3f.h5' % (step+1, b_accuracy4)
	batch_model4.save(Path_MP_CytoMAD +'\\'+filename7)
    
	filename8 = 'cmodel_all_%06d_%.3f.h5' % (step+1, c_accuracy)
	cell_model.save(Path_MP_CytoMAD +'\\'+filename8)

	filename9 = 'bmodelCNN_all_%06d_%.3f.h5' % (step+1, b_CNN_accuracy)
	batch_model_CNN.save(Path_MP_CytoMAD +'\\'+filename9)
    
	filename10 = 'cmodelCNN_all_%06d_%.3f.h5' % (step+1, c_CNN_accuracy)
	cell_model_CNN.save(Path_MP_CytoMAD +'\\'+filename10)

	print('>Saved: %s, %s, %s, %s, %s, %s, %s, %s, %s and %s' % (filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9, filename10))

 
# load and prepare training images
def load_multi_dataset(BasePath, Date, Cells, state, cellsCount):
	FirstTimeIndicator = 0;
    
	filenames = ['_BF.mat', '_QPI.mat']
	DateUnique = list(set(Date))
	BatchDict = {n: DateUnique[n] for n in range(len(DateUnique))}
	CellsUnique = list(set(Cells))
	CellsDict = {n: CellsUnique[n] for n in range(len(CellsUnique))}
    
    # Load and concatenate images data
	for i in range(len(Date)):
		filename = BasePath + '\\' + Date[i] + '\\'+ Cells[i] + '\\' + state + '\\' + Cells[i] + filenames[0]
		Images = mat73.loadmat(filename)
		BF = Images['BF']
		filename = BasePath + '\\' + Date[i] + '\\'+ Cells[i] + '\\' + state + '\\' + Cells[i] + filenames[1]
		Images = mat73.loadmat(filename)
		QPI = Images['QPI']
		BF = np.array(BF, dtype=np.float32)
		QPI = np.array(QPI, dtype=np.float32)
		BF = np.array(BF[:,:,:cellsCount[i]])
		QPI = np.array(QPI[:,:,:cellsCount[i]])
        
        # Create batch and cell type labels
		BatchType = DateUnique.index(Date[i])
		CellType = CellsUnique.index(Cells[i])
		BatchLabel = np.ones((cellsCount[i],1))*BatchType
		CellLabel = np.ones((cellsCount[i],1))*CellType
        
		if FirstTimeIndicator==0:
			BFs = BF
			QPIs = QPI
			BatchLabels = BatchLabel
			CellLabels = CellLabel
			FirstTimeIndicator = 1
		else:
			BFs = np.concatenate((BFs, BF), axis=2)
			QPIs = np.concatenate((QPIs, QPI), axis=2)
			BatchLabels = np.concatenate((BatchLabels, BatchLabel), axis=0)
			CellLabels = np.concatenate((CellLabels, CellLabel), axis=0)
		print(Date[i], ',', Cells[i])
		print('BFs: ', BF.shape)
		print('QPIs: ', QPI.shape)
		print('BatchLabels: ', BatchLabel.shape)
		print('CellLabels: ', CellLabel.shape)
        
	print('Total:')
	print('BFs: ', BFs.shape)
	print('QPIs: ', QPIs.shape)
	print('BatchLabels: ', BatchLabels.shape)
	print('CellLabels: ', CellLabels.shape)
    
    # Shuffle the dataset
	RandomList = list(range(QPIs.shape[2]))
	random.shuffle(RandomList)
	FirstTimeIndicator = 0;
	ImageTotal = QPIs.shape[2]
	ImageWidth = 128
	ImageHeight = 128
	shuffled_BFs_3D = np.zeros((ImageTotal, ImageWidth, ImageHeight))
	shuffled_QPIs_3D = np.zeros((ImageTotal, ImageWidth, ImageHeight))
	shuffled_BatchLabels_index = BatchLabels[RandomList]
	shuffled_CellLabels_index = CellLabels[RandomList]
	TotalBatch = len(DateUnique)
	shuffled_BatchLabels_onehot = np.squeeze(one_hot(shuffled_BatchLabels_index, TotalBatch, axis = 0), axis=2)
	TotalCellType = len(CellsUnique)
	shuffled_CellLabels_onehot = np.squeeze(one_hot(shuffled_CellLabels_index, TotalCellType, axis = 0), axis=2)

	for i in range(len(RandomList)):
		shuffled_BF = BFs[:,:,RandomList[i]]
		shuffled_QPI = QPIs[:,:,RandomList[i]]
		shuffled_BF_128 = cv2.resize(shuffled_BF, dsize=(ImageWidth,ImageHeight), interpolation=cv2.INTER_CUBIC)
		shuffled_QPI_128 = cv2.resize(shuffled_QPI, dsize=(ImageWidth,ImageHeight), interpolation=cv2.INTER_CUBIC)
		shuffled_BF_3D = np.expand_dims(shuffled_BF_128,axis=0)
		shuffled_QPI_3D = np.expand_dims(shuffled_QPI_128,axis=0)
		shuffled_BFs_3D[i,:,:] = shuffled_BF_3D
		shuffled_QPIs_3D[i,:,:] = shuffled_QPI_3D

	shuffled_BFs_4D = np.expand_dims(shuffled_BFs_3D, axis=3)
	shuffled_QPIs_4D = np.expand_dims(shuffled_QPIs_3D, axis=3)
    
	print('BFs 4D Input shuffled: ', shuffled_BFs_4D.shape)
	print('QPIs 4D shuffled: ', shuffled_QPIs_4D.shape)
	print('BatchLabels OneHot shuffled: ', shuffled_BatchLabels_onehot.shape)
	print('CellLabels OneHot shuffled: ', shuffled_CellLabels_onehot.shape)

	X1, X2, C1, C_onehot, C_dict, B1, B_onehot, B_dict = shuffled_BFs_4D, shuffled_QPIs_4D, shuffled_CellLabels_index, shuffled_CellLabels_onehot.T, CellsDict, shuffled_BatchLabels_index, shuffled_BatchLabels_onehot.T, BatchDict

	return [X1, X2, C1, C_onehot, C_dict, B1, B_onehot, B_dict]
    
# train pix2pix models
def train(d_model, g_model, cell_model, cell_model_CNN, batch_model_CNN, gan_model, dataset, dataset_valid, timestr, NoOfBatch, batch_Lrate, batch_input_shape, Path_Figures, Path_Figures_CytoMAD, Path_MP, Path_MP_CytoMAD, n_epochs1=100, n_epochs2=100, n_epochs3=300, epoch_interval = 5, n_batch=10):
	# Determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
    
	# Unpack dataset
	trainA, trainB, train_cell, train_cell_onehot, cell_dict, train_batch, train_batch_onehot, batch_dict = dataset
	testA, testB, test_cell, test_cell_onehot, _, test_batch, test_batch_onehot, _ = dataset_valid
	# Calculate the number of batches per training epoch
	bat_per_epo = int(trainA.shape[0] / n_batch)
	# Calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs1
    # Generate a batch of validation samples
	[X_realA_test, X_realB_test, _, cell_onehot_test, _, batch_onehot_test], y_real_test = generate_real_samples(dataset_valid, n_batch, n_patch)
	g_loss_valid_reference = 10000
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB, _, cell_onehot, _, batch_onehot], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, features_gen, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		features_gen = np.squeeze(features_gen, axis=2)
		features_gen = np.squeeze(features_gen, axis=1)
		# update discriminator with real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator with predicted samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo) == 0:
			g_loss_valid, _, _ = gan_model.test_on_batch(X_realA_test, [y_real_test, X_realB_test])
			if g_loss_valid < g_loss_valid_reference:
				g_model_best = g_model
				d_model_best = d_model
				gan_model_best = gan_model
				summarize_performance(i, g_model, dataset_valid, Path_Figures, Path_MP)
				g_loss_valid_reference = g_loss_valid
    
	g_model = g_model_best
	d_model = d_model_best
	gan_model = gan_model_best
    
	d_model.save(Path_MP + '\\dmodel_all.h5')
    
    # generate real samples for training and validation
	[train_X_realA, _, _, train_cell_onehot, _, train_batch_onehot], y_real = generate_real_samples_all(dataset, n_patch)  
	[test_X_realA, _, _, test_cell_onehot, _, test_batch_onehot], y_real = generate_real_samples_all(dataset_valid, n_patch)
    # predict training and test samples with pre-trained generator
	train_X_fakeB, features_gen, _ = generate_fake_samples(g_model, train_X_realA, n_patch)
	test_X_fakeB, _, _ = generate_fake_samples(g_model, test_X_realA, n_patch)
	features_gen = np.squeeze(features_gen, axis=2)
	features_gen = np.squeeze(features_gen, axis=1)
    
	input_shape = dataset[0].shape[1:]
    
	dataset_valid_tuple_batch = (test_X_fakeB, test_batch_onehot)
	dataset_valid_tuple_cell = (test_X_fakeB, test_cell_onehot)
    
    # train the Neural-Network-based Cell Type Classifier
	cell_model.fit(features_gen, train_cell_onehot, epochs=n_epochs2, validation_split=0.1)
    # train the Convolutional-Neural-Network-based Batch Classifier
	batch_model_CNN.fit(train_X_fakeB, train_batch_onehot, batch_size=500, epochs=n_epochs2, validation_data=dataset_valid_tuple_batch)
    # train the Convolutional-Neural-Network-based Cell Type Classifier
	cell_model_CNN.fit(train_X_fakeB, train_cell_onehot, batch_size=500, epochs=n_epochs2, validation_data=dataset_valid_tuple_cell)
    
	g_loss_valid_reference = 1000000
	g_model_best = g_model
	d_model_best = d_model
    # Calculate the number of batches per training epoch
	bat_per_epo = int(trainA.shape[0] / n_batch)
    # Calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs3
	n_interval = bat_per_epo * epoch_interval
	[X_realA_test, X_realB_test, _, cell_onehot_test, _, batch_onehot_test], y_real_test = generate_real_samples(dataset_valid, n_batch, n_patch)
    
    # generate fake batch label for guiding batch removal
	fake_batch_onehot_test = generate_fake_batchlabel(batch_onehot_test)

	for i in range(n_steps):
        # train new Neural-Network-based Batch Classifier for every 5 epoch
		if i%n_interval == 0:
			g_loss_valid_reference = 1000000
			g_model = g_model_best
			d_model = d_model_best
			batch_model1 = define_classifier_dropout(NoOfBatch, input_shape = batch_input_shape, Lrate = batch_Lrate)
			batch_model2 = define_classifier_dropout(NoOfBatch, input_shape = batch_input_shape, Lrate = batch_Lrate)
			batch_model3 = define_classifier_dropout(NoOfBatch, input_shape = batch_input_shape, Lrate = batch_Lrate)
			batch_model4 = define_classifier_dropout(NoOfBatch, input_shape = batch_input_shape, Lrate = batch_Lrate)
			[X_realA, _, _, cell_onehot, _, batch_onehot], y_real = generate_real_samples_all(dataset, n_patch)  
			_, features_gen, _ = generate_fake_samples(g_model, X_realA, n_patch)
			features_gen = np.squeeze(features_gen, axis=2)
			features_gen = np.squeeze(features_gen, axis=1)
			RandomFeatures = np.random.permutation(features_gen.shape[1])
			RandomFeaturesList1 = RandomFeatures[:batch_input_shape[0]]
			RandomFeaturesList2 = RandomFeatures[batch_input_shape[0]:batch_input_shape[0]*2]
			RandomFeaturesList3 = RandomFeatures[batch_input_shape[0]*2:batch_input_shape[0]*3]
			RandomFeaturesList4 = RandomFeatures[batch_input_shape[0]*3:batch_input_shape[0]*4]
			batch_model1.fit(features_gen[:,RandomFeaturesList1], batch_onehot, epochs=n_epochs2, validation_split=0.1)
			batch_model2.fit(features_gen[:,RandomFeaturesList2], batch_onehot, epochs=n_epochs2, validation_split=0.1)
			batch_model3.fit(features_gen[:,RandomFeaturesList3], batch_onehot, epochs=n_epochs2, validation_split=0.1)
			batch_model4.fit(features_gen[:,RandomFeaturesList4], batch_onehot, epochs=n_epochs2, validation_split=0.1)
            # Define CytoMAD Model
			mapping_model_gan = define_mapping_gan(g_model, d_model, cell_model, cell_model_CNN, batch_model_CNN, batch_model1, batch_model2, batch_model3, batch_model4, RandomFeaturesList1, RandomFeaturesList2, RandomFeaturesList3, RandomFeaturesList4, input_shape)

		# select a batch of real samples
		[X_realA, X_realB, _, cell_onehot, _, batch_onehot], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate fake batch label for guiding batch removal
		fake_batch_onehot = generate_fake_batchlabel(batch_onehot)
		# generate a batch of fake samples
		X_fakeB, features_gen, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		features_gen = np.squeeze(features_gen, axis=2)
		features_gen = np.squeeze(features_gen, axis=1)
        
        # update discriminator with real samples
		_ = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator with predicted samples
		_ = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update CytoMAD model
		gan_loss, d_loss, c_CNN_loss, c_loss, b_CNN_loss, b_loss1, b_loss2, b_loss3, b_loss4, g_loss = mapping_model_gan.train_on_batch(X_realA, [y_real, cell_onehot, cell_onehot, fake_batch_onehot, fake_batch_onehot, fake_batch_onehot, fake_batch_onehot, fake_batch_onehot, X_realB])

		# summarize performance
		[_, features_gen] = g_model.predict(X_realA)
		features_gen = np.squeeze(features_gen, axis=2)
		features_gen = np.squeeze(features_gen, axis=1)
		batch_predict1 = batch_model1.predict(features_gen[:,RandomFeaturesList1])
		batch_predict2 = batch_model2.predict(features_gen[:,RandomFeaturesList2])
		batch_predict3 = batch_model3.predict(features_gen[:,RandomFeaturesList3])
		batch_predict4 = batch_model4.predict(features_gen[:,RandomFeaturesList4])
		cell_predict = cell_model.predict(features_gen)
		batch_CNN_predict = batch_model_CNN.predict(X_fakeB)
		cell_CNN_predict = cell_model_CNN.predict(X_fakeB)
		fake_batch_label = np.argmax(fake_batch_onehot, axis=1)
		cell_label = np.argmax(cell_onehot, axis=1)
		batch_predict1 = np.argmax(batch_predict1, axis=1)
		batch_predict2 = np.argmax(batch_predict2, axis=1)
		batch_predict3 = np.argmax(batch_predict3, axis=1)
		batch_predict4 = np.argmax(batch_predict4, axis=1)
		cell_predict = np.argmax(cell_predict, axis=1)
		batch_CNN_predict = np.argmax(batch_CNN_predict, axis=1)
		cell_CNN_predict = np.argmax(cell_CNN_predict, axis=1)
		print('Batch accuracy: 1. ', accuracy_score(fake_batch_label, batch_predict1), ', 2. ', accuracy_score(fake_batch_label, batch_predict2), ', 3. ', accuracy_score(fake_batch_label, batch_predict3), ', 4. ', accuracy_score(fake_batch_label, batch_predict4))
		print('Batch CNN accuracy:', accuracy_score(fake_batch_label, batch_CNN_predict))
		print('Cell accuracy:', accuracy_score(cell_label, cell_predict))
		print('Cell CNN accuracy:', accuracy_score(cell_label, cell_CNN_predict))
		print('>%d, gan[%.3f] d[%.3f] cCNN[%.3f] c[%.3f] bCNN[%.3f] b1[%.3f] b2[%.3f] b3[%.3f] b4[%.3f] g[%.3f]' % (i+1, gan_loss, d_loss, c_CNN_loss, c_loss, b_CNN_loss, b_loss1, b_loss2, b_loss3, b_loss4, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo) == 0:
			g_loss_valid, _, _, _, _, _, _, _, _, _ = mapping_model_gan.test_on_batch(X_realA_test, [y_real_test, cell_onehot_test, cell_onehot_test, fake_batch_onehot_test, fake_batch_onehot_test, fake_batch_onehot_test, fake_batch_onehot_test, fake_batch_onehot_test, X_realB_test])
			if g_loss_valid < g_loss_valid_reference:
				summarize_performance_all(i, d_model, g_model, cell_model_CNN, cell_model, batch_model_CNN, batch_model1, batch_model2, batch_model3, batch_model4, RandomFeaturesList1, RandomFeaturesList2, RandomFeaturesList3, RandomFeaturesList4, dataset_valid, Path_Figures_CytoMAD, Path_MP_CytoMAD)
				g_loss_valid_reference = g_loss_valid
				g_model_best = g_model
				d_model_best = d_model

###### Base path for loading data and saving results ######
# Data Path
BasePath = '.\\Data\\7LungCancerCellLines'
# Results Path
SavePath = '.\\Results\\7LungCancerCellLines'

###### Load Image Data ######
# Define the batch
Date = ['Batch1','Batch3','Batch4','Batch1','Batch2','Batch5', 'Batch3','Batch6','Batch7', 'Batch2','Batch5','Batch6', 'Batch3','Batch4','Batch6', 'Batch1','Batch3','Batch7', 'Batch1','Batch2','Batch6']
NoOfBatch = len(set(Date))
# Define the cell Type
Cells = ['H69', 'H69', 'H69', 'H358', 'H358', 'H358','H520','H520','H520','H526','H526','H526','H1975','H1975','H1975', 'H2170','H2170','H2170', 'HCC827','HCC827','HCC827']
NoOfCellType = len(set(Cells))
# No of cells per class during training
cellsCount = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
# Load training dataset
state = 'Train'
dataset = load_multi_dataset(BasePath, Date, Cells, state, cellsCount)
print('Loaded', dataset[0].shape, dataset[1].shape)

# No of cells per class in validation
CellsPerCellLines_test = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
# Load validation dataset
state = 'Valid'
dataset_valid = load_multi_dataset(BasePath, Date, Cells, state, CellsPerCellLines_test)
print('Loaded Valid', dataset_valid[0].shape, dataset_valid[1].shape)

###### Define the deep learning models ######
# define input shape based on the loaded dataset
input_shape = dataset[0].shape[1:]
output_shape = dataset[1].shape[1:]

# Define the generator and the discriminator with Pix2Pix architecture
d_model = define_discriminator(input_shape, output_shape, Lrate = 0.0001)
g_model = define_generator(input_shape)
gan_model = define_gan(g_model, d_model, input_shape)

# Define the learning rate and input shape of the Neural-Network-based Batch Classifier 
batch_Lrate = 0.05
batch_input_shape = (128,)
# Define the Neural-Network-based Cell Type Classifier 
cell_model = define_classifier(NoOfCellType, Lrate = 0.001)

# Define the Convolutional-Neural-Network-based Batch Classifier and Cell Type Classifier
batch_model_CNN = define_CNN(output_shape, NoOfBatch, Lrate = 0.001)
cell_model_CNN = define_CNN(output_shape, NoOfCellType, Lrate = 0.001)

###### Create folders for saving the training models and results ######
# Create path for saving resultant figures
timestr = time.strftime("%Y%m%d_%H%M%S")
BasePath_Figures = SavePath+'\\Figures'
if not os.path.exists(BasePath_Figures):
    os.mkdir(BasePath_Figures)
Path_Figures = BasePath_Figures + '\\' + timestr + '_Pretraining_CytoMAD_With_Image_Coversion'
os.mkdir(Path_Figures)
Path_Figures_CytoMAD = BasePath_Figures + '\\' + timestr + '_CytoMAD_With_Batch_Removal'
os.mkdir(Path_Figures_CytoMAD)

# Create path for saving trained models
BasePath_MP = SavePath+'\\ModelParameters'
if not os.path.exists(BasePath_MP):
    os.mkdir(BasePath_MP)
Path_MP = BasePath_MP + '\\' + timestr + '_Pretraining_CytoMAD_With_Image_Coversion'
os.mkdir(Path_MP)
Path_MP_CytoMAD = BasePath_MP + '\\' + timestr + '_CytoMAD_With_Batch_Removal'
os.mkdir(Path_MP_CytoMAD)

###### Start training ######
train(d_model, g_model, cell_model, cell_model_CNN, batch_model_CNN, gan_model, dataset, dataset_valid, timestr, NoOfBatch, batch_Lrate, batch_input_shape, Path_Figures, Path_Figures_CytoMAD, Path_MP, Path_MP_CytoMAD)
