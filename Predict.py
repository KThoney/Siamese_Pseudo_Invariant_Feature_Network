import utils
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import mat73
from tensorflow.keras.optimizers import *

root_dir = 'E:/py_codes/RRN_Deeplearning/Training data/Full_data'
ref_dir = root_dir + '/Reference'
sen_dir = root_dir + '/Sensed'

ref_dataset = glob.glob(ref_dir+'/*.mat')	# the number of classes
ref_classes = os.listdir(ref_dir)
ref_classes.sort(key=len)
os.chdir(ref_dir)
Data_ref = np.zeros((len(ref_dataset),config.IMG_SHAPE,config.IMG_SHAPE,3))

for i in range(len(ref_dataset)):
	file_name = ref_classes[i]
	print(file_name)
	temp = mat73.loadmat(file_name)['Ref_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,3)
	temp_data = temp_data[:,:,:]
	Data_ref[i] = temp_data  # ch01

sen_dataset = glob.glob(sen_dir+'/*.mat')	# the number of classes
sen_classes = os.listdir(sen_dir)
sen_classes.sort(key=len)
os.chdir(sen_dir)
Data_sen = np.zeros((len(sen_dataset),config.IMG_SHAPE,config.IMG_SHAPE,3))

for i in range(len(sen_dataset)):
	file_name = sen_classes[i]
	print(file_name)
	temp = mat73.loadmat(file_name)['Sen_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,3)
	temp_data = temp_data[:,:,:]
	Data_sen[i] = temp_data  # ch01

Data_ref, Data_sen = utils.data_resize(Data_ref,Data_sen,64)

from Models.SiamesePIFsnet import *
# configure the siamese network
print("[INFO] building siameseSAE network...")
SIPIF_net = SIPIF_net((64,64,3),(64,64,3))
# compile the model
print("[INFO] compiling model...")
SIPIF_net.compile(optimizer=Adam(learning_rate=1e-04), loss='binary_crossentropy', metrics=['accuracy'])# train the model
print("[INFO] Predicting model...")
check_dir = "E:/py_codes/RRN_Deeplearning/SiamSAE_PIFs/output/SIPIF_net/cp.ckpt"
SIPIF_net.load_weights(check_dir)
results=SIPIF_net.predict(([Data_ref, Data_sen]))
pred_data = pd.DataFrame(results)
output_file = 'E:/py_codes/RRN_Deeplearning/Training data/Full_data/SIPIF_net_pred2.csv'
pred_data.to_csv(output_file, header=False, index=False)
