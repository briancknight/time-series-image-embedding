import read_airbus_data
import numpy as np
import pywt
from matplotlib import pyplot as plt
import time

t_data, v_data = read_airbus_data.read_airbus_data()


def gray_scale_encoding(X, params={}):
	if params == {}:
		# default parameters
		params = {
		's':8, 
		'P':255, 
		'K': 64, 
		'UB': np.max(X), 
		'LB': np.min(X),
		'round':True }

	# assign parameters
	s = params['s']
	P = params['P']
	K = params['K']
	UB = params['UB']
	LB = params['LB']
	C = 1/(UB - LB)
	im = np.zeros((K,K))

	if params['round']: # round
		for i in range(K):
			for j in range(K):
				im[i,j] = np.round(
					P*(X[(i - 1)*s + j] - LB)*C)

	else: # don't round
		for i in range(K):
			for j in range(K):
				im[i,j] = P*(X[(i - 1)*s + j] - LB)*C

	return im

def main():
	print(np.shape(t_data))

	UB = 1.2*np.max(t_data)
	LB = 1.2*np.min(t_data)
	K = 64
	print(UB, LB)
	test = gray_scale_encoding(
		t_data[0][:512], params={'s':7, 'P':1, 'K':K, 'UB':UB,'LB':LB})

	plt.imshow(test,cmap='jet')
	plt.show()

	# turn training data into a sequence of 120 images:
	params = {'s':7, 'P':1, 'K':K, 'UB':UB,'LB':LB}
	gs_t_data = np.zeros((120, K, K))
	t0 = time.time()
	for i in range(120):
		start = int(512*i)
		stop = int(512*(i+1))
		gs_t_data[i] = gray_scale_encoding(t_data[0][start:stop])

	print('time elapsed is: ', time.time() - t0)
if __name__ == '__main__':
	main()