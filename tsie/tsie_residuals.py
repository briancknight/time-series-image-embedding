import numpy as np
import pandas as pd
import tensorflow as tf

def get_residual_distribution(model, t_data):
	"""
	Takes autoencoder and (transformed) training data and computes all residuals 
	args: 
		model: keras model for computing reconstructions 
		(k, m, m, 1) -> (k, m, m, 1)

		t_data: numpy array of transformed training data with shape 
		np.shape(t_data) = (N, l, m, m, 1) = (N, 120, 64, 64 ,1)

	returns: residuals, a numpy array of shape (N, l, 1) = (N*120, 1),
		     containing residuals Res_k = ||x_or - x_rc||_2
	"""

	N = np.shape(t_data)[0]
	l = 120
	m = 64

	residuals = np.zeros((N*l,1))
	all_t_images = np.reshape(t_data, (N*l, 64, 64, 1))
	rc = model.predict(all_t_images)

	residuals = np.linalg.norm(all_t_images - rc, axis = (1,2))

	return residuals


def classify_ts_residuals(model, data, q99):
	""" data should shape (N, l, m, m, 1) = (N, 120, 64, 64 ,1)
	# model should be autoencoder (m, m, 1) -> (m, m, 1)
	returns residuals: np array of shape (N, l, 1) = (N, 120, 1)
			containing residual of each reconstruction in the series
			anomalous: binary array which is 0 if the max residual
			is below the 99th percentile, and 1 otherwise.
	"""
	N = np.shape(data)[0]
	l = 120
	m = 64
	residuals = np.zeros((N, l, 1))
	anomalous = np.zeros(N)
	
	for i in range(N):
		seq = data[i]
		rc = model.predict(seq)
		# compute individual residuals in (120, 1) array:
		residuals[i] = np.linalg.norm(np.reshape(seq - rc, (l, m**2, 1)), axis = 1)

		if max(residuals[i]) > q99:
			anomalous[i] = 1

	return residuals, anomalous


def get_residual_distribution_1D(model, t_data):
	"""
	Takes autoencoder and (transformed) training data and computes all residuals 
	args: 
		model: keras model for computing reconstructions 
		(k, m, 1) -> (k, m, 1)

		t_data: numpy array of transformed training data with shape 
		np.shape(t_data) = (N, l, m, 1) = (N, 120, 512, 1)

	returns: residuals, a numpy array of shape (N, l, 1) = (N*120, 1),
		     containing residuals Res_k = ||x_or - x_rc||_2
	"""

	N = np.shape(t_data)[0]
	l = 120
	m = 512

	residuals = np.zeros((N*l,1))
	all_t_images = np.reshape(t_data, (N*l, m, 1))
	rc = model.predict(all_t_images)

	residuals = np.linalg.norm(all_t_images - rc, axis = 1)

	return residuals


def classify_ts_residuals_1D(model, data, q99):
	""" data should shape (N, l, m, 1) = (N, 120, 512, 1)
	# model should be autoencoder (m, 1) -> (m, 1)
	returns residuals: np array of shape (N, l, 1) = (N, 120, 1)
			containing residual of each reconstruction in the series
			anomalous: binary array which is 0 if the max residual
			is below the 99th percentile, and 1 otherwise.
	"""
	N = np.shape(data)[0]
	l = 120
	m = 512
	residuals = np.zeros((N, l, 1))
	anomalous = np.zeros(N)
	
	for i in range(N):
		seq = data[i]
		rc = model.predict(seq)
		# compute individual residuals in (120, 1) array:
		residuals[i] = np.linalg.norm(seq-rc, axis = 1)

		if max(residuals[i]) > q99:
			anomalous[i] = 1

	return residuals, anomalous


##################################################

def results_of_gs_mm_encoding():

	# load 2D CAE
	# model = tf.keras.models.load_model('tsie_2D_CAE_gs')

	# load training data & construct labels (no anomalous sequences)
	# gs_p1_t_data = np.reshape(np.load('../airbus_data/gs_p1_t_data.npy'), (1677, 120, 64, 64, 1))

	# compute reconstructions and residuals
	#residuals = get_residual_distribution(model, gs_p1_t_data)
	residuals = np.load('residual_data/gs_p1_t_residuals.npy')

	q99 = np.percentile(residuals, 99)
	# np.save('residual_data/gs_p1_t_residuals.npy',residuals)

	# load validation data & validation labels
	# gs_p1_v_data = np.reshape(np.load('../airbus_data/gs_p1_v_data.npy'), (594, 120, 64, 64, 1))

	df = pd.read_csv('../airbus_data/dfvalid_groundtruth.csv')
	v_labels = np.array(df['anomaly'])

	# v_residuals, anomalous = classify_ts_residuals(model, gs_p1_v_data, q99)
	# np.save('residual_data/gs_p1_v_residuals.npy', v_residuals)
	v_residuals = np.load('residual_data/gs_p1_v_residuals.npy')
	anomalous = [np.max(v_residuals[i]) > q99 for i in range(594)]

	accuracy = 1 - np.sum(np.abs(anomalous - v_labels))/len(v_labels)

	print('accuracy is: ', accuracy)
	TPR = sum(anomalous + v_labels == 2)/297
	FPR = sum(v_labels - anomalous == -1 )/297
	print('TPR % :', TPR)
	print('FPR % :', FPR)

def results_of_gs_p1round_encoding():

	# load 2D CAE
	# model = tf.keras.models.load_model('tsie_2D_CAE_gs_p1round')

	# load training data & construct labels (no anomalous sequences)
	# t_data = np.reshape(np.load('../airbus_data/gs_p1round_t_data.npy'), (1677, 120, 64, 64, 1))

	# compute reconstructions and residuals
	# residuals = get_residual_distribution(model, t_data)
	# np.save('residual_data/gs_p1round_t_residuals.npy', residuals)
	residuals = np.load('residual_data/gs_p1round_t_residuals.npy')
	q99 = np.percentile(residuals, 99)

	# load validation data & validation labels
	# v_data = np.reshape(np.load('../airbus_data/gs_p1round_v_data.npy'), (594, 120, 64, 64, 1))

	df = pd.read_csv('../airbus_data/dfvalid_groundtruth.csv')
	v_labels = np.array(df['anomaly'])

	# v_residuals, anomalous = classify_ts_residuals(model, v_data, q99)
	# np.save('residual_data/gs_p1round_v_residuals.npy', v_residuals)
	v_residuals = np.load('residual_data/gs_p1round_v_residuals.npy')
	print(np.shape(v_residuals))
	anomalous = np.zeros(594) #[np.max(v_residuals[i]) > q99 for i in range(594)]
	for i in range(594):
		if np.max(v_residuals[i]) > q99:
			anomalous[i] = 1

	accuracy = 1 - np.sum(np.abs(anomalous - v_labels))/len(v_labels)

	print('accuracy is: ', accuracy)
	TPR = sum(anomalous + v_labels == 2)/297
	FPR = sum(v_labels - anomalous == -1 )/297
	print('TPR % :', TPR)
	print('FPR % :', FPR)

def results_of_no_encoding():

	# load model, validation data, and training residuals:

	CAE_1D = tf.keras.models.load_model('tsie_1D_CAE')
	v_data = np.array(np.reshape(np.array(pd.read_hdf('../airbus_data/dfvalid.h5')), (594, 120, 512, 1)))
	residuals_1D = np.load('residual_data/t_residuals_1D.npy')
	
	# compute 99th percentile
	q99_1D = np.percentile(residuals_1D, 99)

	# load true labels:
	df = pd.read_csv('../airbus_data/dfvalid_groundtruth.csv')
	v_labels = np.array(df['anomaly'])

	v_residuals_1D, anomalous_1D = classify_ts_residuals_1D(CAE_1D, v_data, q99_1D)
	np.save('residual_data/v_residuals_1D', v_residuals_1D)

	accuracy_1D = 1 - (np.sum(np.abs(anomalous_1D - v_labels))) / len(v_labels)
	print('accuracy is: ', accuracy_1D) # 84% 

	TPR_1D = sum(anomalous_1D + v_labels == 2)/297 # 68% 
	FPR_1D = sum(v_labels - anomalous_1D == -1)/297 # 0%

	print('TPR is: ', TPR_1D)
	print('FPR is: ', FPR_1D)

def main():

	#results_of_no_encoding()
	results_of_gs_p1round_encoding()
	results_of_gs_mm_encoding()

if __name__ == '__main__':
	main()