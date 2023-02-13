import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def show_image(image1, image2):
	plt.figure(figsize=(8, 4))
	plt.grid(False)
	plt.subplot(1, 2, 1)
	plt.imshow(image1)
	plt.subplot(1, 2, 2)
	plt.imshow(image2)
	plt.show()

def display_images(left, right, predictions, labels, title, n):
	plt.figure(figsize=(17,3))
	plt.title(title)
	plt.yticks([])
	plt.xticks([])
	plt.grid(None)
	left = np.reshape(left, [n, 28, 28])
	left = np.swapaxes(left, 0, 1)
	left = np.reshape(left, [28, 28*n])
	plt.imshow(left)
	plt.figure(figsize=(17,3))
	plt.yticks([])
	plt.xticks([28*x+14 for x in range(n)], predictions)
	for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
		if predictions[i] > 0.5: t.set_color('red')
		plt.grid(None)
		right = np.reshape(right, [n, 28, 28])
		right = np.swapaxes(right, 0, 1)
		right = np.reshape(right, [28, 28*n])
		plt.imshow(right)


def augment(data, number, patch_size):
	scale = np.linspace(2, 8, 4)
	sample, row, col, band = data.shape
	number_set = number - 1
	idx = 1
	with tf.device('/device:cpu:0'):
		for i in range(number_set):
			scale_idx = np.random.choice(len(scale), 1, False)
			patch_st = int(0 + scale[scale_idx[0]])
			patch_end = int(row - scale[scale_idx[0]])
			resam_data = data[:, patch_st:patch_end, patch_st:patch_end, :]
			resam_data2 = tf.image.resize(resam_data, [patch_size, patch_size], method='bicubic')
			resam_data3 = tf.image.random_contrast(resam_data2, 0.3, 0.9)
			agu_data = tf.image.rot90(resam_data3, k=idx)
			idx = idx + 1
			if i == 0:
				new_data = tf.image.resize(data, [patch_size, patch_size], method='bicubic')
				augumentation_data = tf.concat([new_data, agu_data], 0)
			else:
				augumentation_data = tf.concat([augumentation_data, agu_data], 0)

		augumentation_data = augumentation_data.numpy()

	return augumentation_data


def data_optimizer(ref_data, sen_data, label):
	sample, row, col, band = ref_data.shape
	inlier_idx = np.where(label[:, 0] == 1)
	inlier_idx = inlier_idx[0]
	outlier_idx = np.where(label[:, 0] == 0)
	outlier_idx = outlier_idx[0]

	new_ref = np.zeros((min([len(inlier_idx), len(outlier_idx)]) * 2, row, col, band))
	new_sen = np.zeros((min([len(inlier_idx), len(outlier_idx)]) * 2, row, col, band))
	new_label = np.zeros((min([len(inlier_idx), len(outlier_idx)]) * 2, 1))
	n = 0
	m = 0

	for i in range(len(new_label)):
		if i % 2 == 0:
			new_ref[i, :, :, :] = ref_data[inlier_idx[n], :, :, :]
			new_sen[i, :, :, :] = sen_data[inlier_idx[n], :, :, :]
			new_label[i] = 1
			n = n + 1


		else:
			new_ref[i, :, :, :] = ref_data[outlier_idx[m], :, :, :]
			new_sen[i, :, :, :] = sen_data[outlier_idx[m], :, :, :]
			new_label[i] = 0
			m = m + 1

	return new_ref, new_sen, new_label

def data_resize(ref_data, sen_data, patch_size):
	with tf.device('/device:cpu:0'):
		new_ref_data=tf.image.resize(ref_data, [patch_size, patch_size], method='bicubic')
		new_sen_data=tf.image.resize(sen_data, [patch_size, patch_size], method='bicubic')
		new_ref_data=new_ref_data.numpy()
		new_sen_data=new_sen_data.numpy()

	return new_ref_data, new_sen_data

def compute_accuracy(y_true, y_pred):
	pred = y_pred.ravel() < 0.5
	return np.mean(pred == y_true)

def splitTrainTestSet(X1, X2, y, testRatio, randomState=345):
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)

    return X1_train, X1_test, X2_train, X2_test, y_train, y_test


def plot_model_loss(train_loss, val_loss):
	plt.figure()
	plt.plot(train_loss, color='blue', label='Train_loss')
	plt.plot(val_loss, color='red', label='Validation_loss')
	plt.grid(False)
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(('Train_loss', 'Validation_loss'))
	ax = plt.gca()
	ax.set_facecolor('white')
	ax.spines['left'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	plt.show()


def plot_model_acc(train_acc, val_acc):
	plt.figure()
	plt.plot(train_acc, label="Train_accuracy", color='blue')
	plt.plot(val_acc, label="Validation_accuracy", color='red')
	plt.title("Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend(('Train_accuracy', 'Validation_accuracy'))
	ax = plt.gca()
	ax.set_facecolor('white')
	ax.spines['left'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	plt.show()

def save_model_loss(train_loss, val_loss, output_dir):
	train_loss1 = train_loss
	val_loss2 = val_loss
	model_loss = np.zeros((30, 2))

	model_loss[:, 0] = train_loss1
	model_loss[:, 1] = val_loss2

	loss_data = pd.DataFrame(model_loss)
	output_file = output_dir + 'model_loss.csv'
	loss_data.to_csv(output_file, header=False, index=False)


def save_model_acc(train_acc, val_acc, output_dir):
	train_acc1 = train_acc
	val_acc2 = val_acc
	model_acc = np.zeros((30, 2))

	model_acc[:, 0] = train_acc1
	model_acc[:, 1] = val_acc2

	acc_data = pd.DataFrame(model_acc)
	output_file = output_dir + 'model_acc.csv'
	acc_data.to_csv(output_file, header=False, index=False)

