import numpy as np
import pandas as pd
import time
# import the necessary tf packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class ConvAutoencoder1D:
	@staticmethod
	def build(width, height, filters=(64, 128, 256), latentDim=300):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (width, height)
		chanDim = -1
		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs
		# loop over the number of filters
		for f in filters:
			if f == 64:
				stride = 4
			else:
				stride = 2
			# apply a CONV => RELU => BN operation
			x = Conv1D(f, kernel_size=int(1024/f), strides=stride, padding="same")(x) # 2, 2 from paper
			x = AveragePooling1D(pool_size=2, strides=2)(x)
			x = LeakyReLU(alpha=0.3)(x) # alpha = 0.4 from paper
			x = BatchNormalization(axis=chanDim)(x) # maybe remove this

		# flatten the network and then construct our latent vector of size 300
		volumeSize = K.int_shape(x)
		print(volumeSize)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")

		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2]))(x)

		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			if f == 64:
				stride = 4
			else:
				stride = 2
			x = Conv1DTranspose(f, kernel_size=int(1024/f), strides=stride, padding="same")(x)
			x = UpSampling1D(size=2)(x)
			x = LeakyReLU(alpha=0.3)(x)
			x = BatchNormalization(axis=chanDim)(x)

		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv1DTranspose(height, kernel_size=2, padding="same")(x)
		outputs = Activation("sigmoid")(x)
		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")
		# our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")
		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)

def main():

	# build tf model:
	model = tf.keras.Sequential
	(encoder, decoder, autoencoder) = ConvAutoencoder1D.build(512, 1)
	
	print(encoder.summary())
	print(decoder.summary())
	print(autoencoder.summary())

	# training data for CAE:

	t_data = np.reshape(np.array(pd.read_hdf('../airbus_data/dftrain.h5')), (1677, 120, 512, 1))
	
	trainX = np.reshape(t_data, (1677*120, 512, 1))
	opt = Adam(learning_rate=1e-3)
	autoencoder.compile(loss="mse", optimizer=opt)

	EPOCHS = 10
	BS = 200

	t0 = time.time()
	H = autoencoder.fit(
		trainX, trainX,
		#validation_data=(testX, testX),
		epochs=EPOCHS,
		batch_size=BS)

	print('time elapsed: ', time.time() - t0)

	autoencoder.save('tsie_1D_CAE')
if __name__ == '__main__':
	main()