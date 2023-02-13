import utils
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *

root_dir = 'E:/py_codes/RRN_Deeplearning/Training data/'
ref_dir = root_dir + '/Reference'
sen_dir = root_dir + '/Sensed'
Label_dir =root_dir + '/Label'

ref_dataset = glob.glob(ref_dir+'/*.mat')	# the number of classes
ref_classes = os.listdir(ref_dir)
ref_classes.sort(key=len)
os.chdir(ref_dir)
Data_ref = np.zeros((len(ref_dataset),16,16,4))

for i in range(len(ref_dataset)):
	file_name = ref_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['Ref_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,config.Band)
	temp_data = temp_data[:,:,:]
	Data_ref[i] = temp_data  # ch01

sen_dataset = glob.glob(sen_dir+'/*.mat')	# the number of classes
sen_classes = os.listdir(sen_dir)
sen_classes.sort(key=len)
os.chdir(sen_dir)
Data_sen = np.zeros((len(sen_dataset),config.IMG_SHAPE,config.IMG_SHAPE,4))

for i in range(len(sen_dataset)):
	file_name = sen_classes[i]
	print(file_name)
	temp = sio.loadmat(file_name)['Sen_patch']
	temp_data = temp.reshape(config.IMG_SHAPE, config.IMG_SHAPE,config.Band)
	temp_data = temp_data[:,:,:]
	Data_sen[i] = temp_data  # ch01

os.chdir(Label_dir)
Data_Label = np.zeros(len(sen_dataset))
Label_classes = os.listdir(Label_dir)
temp = pd.read_csv(Label_classes[0], sep=',',header=None)
Data_Label=temp.values.reshape(len(sen_dataset),1)

Data_ref, Data_sen = utils.data_resize(Data_ref,Data_sen,64)
Ref_train, Ref_val, Sen_train, Sen_val, Label_train, Label_val = utils.splitTrainTestSet(Data_ref, Data_sen,Data_Label, 0.4, randomState=345)

# Data agumentation
Ref_train = utils.augment(Ref_train,10,64)
Sen_train = utils.augment(Sen_train,10,64)
Label_train = np.tile(Label_train,(10,1))
Ref_train, Sen_train, Label_train=utils.data_optimizer(Ref_train , Sen_train , Label_train)

from Models.SiamesePIFsnet import *
# configure the siamese network
print("[INFO] building siameseSAE network...")
SIPIF_net = SIPIF_net((64,64,4),(64,64,4))
# compile the model
print("[INFO] compiling model...")
SIPIF_net.compile(optimizer=Adam(learning_rate=5e-03), loss='binary_crossentropy', metrics=['accuracy'])# train the model
print("[INFO] training model...")
check_dir = "E:/py_codes/RRN_Deeplearning/SiamSAE_PIFs/output/SIPIF_net/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_dir, monitor='val_accuracy', mode='max', save_weights_only=True, verbose=1)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=0, mode='max', min_delta=0.005, min_lr=5e-05)
history = SIPIF_net.fit([Ref_train, Sen_train], Label_train,validation_data=([Ref_val, Sen_val], Label_val), batch_size=32,
						epochs=30, verbose=1, shuffle=True, callbacks=[cp_callback, lr_callback])

_, acc = SIPIF_net.evaluate([Ref_val, Sen_val], Label_val)
print("Accuracy is = ", (acc * 100.0), "%")
result=SIPIF_net.predict([Ref_val, Sen_val])

val_loss=history.history["val_loss"]
train_loss=history.history["loss"]
val_acc=history.history["val_accuracy"]
train_acc=history.history["accuracy"]

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn import preprocessing
binary=preprocessing.binarize(result, threshold=0.5)
confusion_matrix(binary,Label_val,labels=[1,0])
p =  precision_score(binary,Label_val)
r =  recall_score(binary,Label_val)
f1 = f1_score(binary,Label_val)
k = cohen_kappa_score(binary,Label_val)
a = accuracy_score(binary,Label_val)
print("Overall accuracy= ", (a * 100.0),"%","\nPrecision = ", p,
	  "\nRecall = ", r, "\nF1-Score = ", f1, "\nKappa = ", k)

