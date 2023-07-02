# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:08:24 2019

@author: Kevin Tsia
"""
# example of CytoMAD with image contrast conversion from single-cell brightfield to QPI, together with batch removal
from tensorflow.keras.models import load_model
import scipy.io
import numpy as np
import time
import os
import mat73
import math
from tensorflow import one_hot

# load and prepare test images
def load_multi_dataset_all(BasePath, Date, Cells, Samples, state):
	FirstTimeIndicator = 0;

	filenames = ['_BF.mat', '_QPI.mat']
	DateUnique = list(set(Date))
	BatchDict = {n: DateUnique[n] for n in range(len(DateUnique))}
	CellsUnique = list(set(Cells))
	CellsDict = {n: CellsUnique[n] for n in range(len(CellsUnique))}
	SamplesUnique = list(set(Samples))
	DrugDict = {n: SamplesUnique[n] for n in range(len(SamplesUnique))}
    
    # Load and concatenate images data
	for i in range(len(Date)):
		filename = BasePath + '\\' + Cells[i] + '\\' + Date[i] + '\\'+ Samples[i] + '\\' + state + '\\' + Samples[i] + filenames[0]
		Images = mat73.loadmat(filename)
		BF = Images['BF']
		filename = BasePath + '\\' + Cells[i] + '\\' + Date[i] + '\\'+ Samples[i] + '\\' + state + '\\' + Samples[i] + filenames[1]
		Images = mat73.loadmat(filename)
		QPI = Images['QPI']
		BF = np.array(BF)
		QPI = np.array(QPI)
		cellsCount = QPI.shape[2]
        
        # Create batch and cell type labels
		BatchType = DateUnique.index(Date[i])
		CellType = CellsUnique.index(Cells[i])
		DrugType = SamplesUnique.index(Samples[i])
        
		BatchLabel = np.ones((cellsCount,1))*BatchType
		CellLabel = np.ones((cellsCount,1))*CellType
		DrugLabel = np.ones((cellsCount,1))*DrugType
        
		if FirstTimeIndicator==0:
			BFs = BF
			QPIs = QPI
			BatchLabels = BatchLabel
			CellLabels = CellLabel
			DrugLabels = DrugLabel
			FirstTimeIndicator = 1
		else:
			BFs = np.concatenate((BFs, BF), axis=2)
			QPIs = np.concatenate((QPIs, QPI), axis=2)
			BatchLabels = np.concatenate((BatchLabels, BatchLabel), axis=0)
			CellLabels = np.concatenate((CellLabels, CellLabel), axis=0)
			DrugLabels = np.concatenate((DrugLabels, DrugLabel), axis=0)
		print(Date[i], ',', Cells[i], ',', Samples[i])
		print('BFs: ', BFs.shape)
		print('QPIs: ', QPIs.shape)
		print('BatchLabels: ', BatchLabels.shape)
		print('CellLabels: ', CellLabels.shape)
		print('DrugLabels: ', DrugLabels.shape)
        
	print('Total:')
	print('BFs: ', BFs.shape)
	print('QPIs: ', QPIs.shape)
	print('BatchLabels: ', BatchLabels.shape)
	print('CellLabels: ', CellLabels.shape)
	print('DrugLabels: ', DrugLabels.shape)

    # Shuffle the dataset
	FirstTimeIndicator = 0;
	ImageTotal = QPIs.shape[2]
	ImageWidth = 128
	ImageHeight = 128
	shuffled_BFs_3D = np.zeros((ImageTotal, ImageWidth, ImageHeight))
	shuffled_QPIs_3D = np.zeros((ImageTotal, ImageWidth, ImageHeight))
	shuffled_BatchLabels_index = BatchLabels
	shuffled_CellLabels_index = CellLabels
	shuffled_DrugLabels_index = DrugLabels
	TotalBatch = len(DateUnique)
	shuffled_BatchLabels_onehot = np.squeeze(one_hot(shuffled_BatchLabels_index, TotalBatch, axis = 0), axis=2)
	TotalCellType = len(CellsUnique)
	shuffled_CellLabels_onehot = np.squeeze(one_hot(shuffled_CellLabels_index, TotalCellType, axis = 0), axis=2)
	TotalDrugType = len(SamplesUnique)
	shuffled_DrugLabels_onehot = np.squeeze(one_hot(shuffled_DrugLabels_index, TotalDrugType, axis = 0), axis=2)  

	for i in range(QPIs.shape[2]):
		shuffled_BF = BFs[:,:,i]
		shuffled_QPI = QPIs[:,:,i]
		shuffled_BF_3D = np.expand_dims(shuffled_BF,axis=0)
		shuffled_QPI_3D = np.expand_dims(shuffled_QPI,axis=0)
		shuffled_BFs_3D[i,:,:] = shuffled_BF_3D
		shuffled_QPIs_3D[i,:,:] = shuffled_QPI_3D

	shuffled_BFs_4D = np.expand_dims(shuffled_BFs_3D, axis=3)
	shuffled_QPIs_4D = np.expand_dims(shuffled_QPIs_3D, axis=3)
    
	print('BFs 4D Input shuffled: ', shuffled_BFs_4D.shape)
	print('QPIs 4D shuffled: ', shuffled_QPIs_4D.shape)
	print('BatchLabels OneHot shuffled: ', shuffled_BatchLabels_onehot.shape)
	print('CellLabels OneHot shuffled: ', shuffled_CellLabels_onehot.shape)
	print('DrugLabels OneHot shuffled: ', shuffled_DrugLabels_onehot.shape)

	X1, X2, D1, D_onehot, D_dict, C1, C_onehot, C_dict, B1, B_onehot, B_dict = shuffled_BFs_4D, shuffled_QPIs_4D, shuffled_DrugLabels_index, shuffled_DrugLabels_onehot.T, DrugDict, shuffled_CellLabels_index, shuffled_CellLabels_onehot.T, CellsDict, shuffled_BatchLabels_index, shuffled_BatchLabels_onehot.T, BatchDict

	return [X1, X2, D1, D_onehot, D_dict, C1, C_onehot, C_dict, B1, B_onehot, B_dict]

# load dataset
BasePath = '.\\Data\\LungCancerDrug'
SavePath = '.\\Results\\LungCancerDrug'

Date = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
Cells = ['Docetaxel', 'Docetaxel', 'Afatinib', 'Afatinib', 'Gemcitabine', 'Gemcitabine']
Samples = ['D000376','D000376','A0019','A0019','G000751','G000751']
state = 'Train'

# load saved CytoMAD model
modelpath = '\\\Alphahku-htc6\\e\\Michelle\\beGAN\\Results\\LungCancerDrug\\ModelParameters'
modelfolder = 'XXXXXXXX_XXXXXX_CytoMAD_With_Batch_Removal'
modelname = 'CytoMADmodel_XXXXXX.h5'
g_model = load_model(modelpath + '\\' + modelfolder + '\\' + modelname)

# Create folders for saving predicted test data
BasePath_TestData = SavePath+'\\TestData'
if not os.path.exists(BasePath_TestData):
    os.mkdir(BasePath_TestData)
Path_TestData = BasePath_TestData + '\\' + modelfolder
os.mkdir(Path_TestData)

for i in range(len(Date)):
    Batch = [Date[i]]
    Cell = [Cells[i]]
    Drug = [Samples[i]]
    # load test data
    print('Loading: ', Cell[0])
    [X1, X2, test_drug, test_drug_onehot, drug_dict, test_cell, test_cell_onehot, cell_dict, test_batch, test_batch_onehot, batch_dict] = load_multi_dataset_all(BasePath, Batch, Cell, Drug, state)
    print('Loaded', X1.shape, X2.shape)
    src_image, tar_image = X1, X2

    [gen_image, gen_features] = g_model.predict(src_image)
    src_image = np.float32(src_image)
    tar_image = np.float32(tar_image)
    gen_features = np.squeeze(gen_features, axis=2)
    gen_features = np.squeeze(gen_features, axis=1)
    
    # save CytoMAD prediction
    print('Saving: ',  Batch[0] , ' ', Cell[0])
    scipy.io.savemat(Path_TestData + '\\CytoMAD_test_'+modelname[:-3]+'_'+Batch[0]+'_'+Cell[0]+'.mat',mdict={'CytoMAD_image': gen_image, 'CytoMAD_features': gen_features})
    scipy.io.savemat(Path_TestData + '\\CytoMAD_test_'+modelname[:-3]+'_'+Batch[0]+'_'+Cell[0]+'_QPI.mat',mdict={'QPI': tar_image})
    scipy.io.savemat(Path_TestData + '\\CytoMAD_test_'+modelname[:-3]+'_'+Batch[0]+'_'+Cell[0]+'_BF.mat',mdict={'BF': src_image})
