import numpy as np
import time
# import the necessary tf packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
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

class ConvAutoencoder:
	@staticmethod
	def build(width, height, depth, filters=(64, 128), latentDim=300):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (width, height, depth)
		chanDim = -1

		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs
		# loop over the number of filters
		for f in filters:
			# apply a CONV => RELU => BN operation
			x = Conv2D(f, kernel_size=2, strides=2, padding="same")(x) # 2, 2 from paper
			x = LeakyReLU(alpha=0.3)(x) # alpha = 0.4 from paper
			x = BatchNormalization(axis=chanDim)(x) # maybe remove this

		# flatten the network and then construct our latent vector of size 300
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")

		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(f, kernel_size=2, strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.3)(x)
			x = BatchNormalization(axis=chanDim)(x)

		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv2DTranspose(depth, kernel_size=2, padding="same")(x)
		outputs = Activation("sigmoid")(x)
		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")
		# our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")
		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)


def main():

	(encoder, decoder, autoencoder) = ConvAutoencoder.build(64, 64, 1)
	
	print(encoder.summary())
	print(decoder.summary())
	print(autoencoder.summary())

	# gray-scale p = 1 transfomred training data for CAE:

	# gs_p1_t_data = np.load('../airbus_data/gs_p1_t_data.npy')
	
	# trainX = np.reshape(gs_p1_t_data, (1677*120, 64, 64, 1))
	# opt = Adam(learning_rate=1e-3)
	# autoencoder.compile(loss="mse", optimizer=opt)

	# EPOCHS = 10
	# BS = 200

	# t0 = time.time()
	# H = autoencoder.fit(
	# 	trainX, trainX,
	# 	#validation_data=(testX, testX),
	# 	epochs=EPOCHS,
	# 	batch_size=BS)

	# print('time elapsed: ', time.time() - t0)

	# autoencoder.save('tsie_2D_CAE_gs')

	# gray-scale p = 1, rounded, transformed training data for CAE:

	gs_p1round_t_data = np.load('../airbus_data/gs_p1round_t_data.npy')
	
	trainXround = np.reshape(gs_p1round_t_data, (1677*120, 64, 64, 1))
	opt = Adam(learning_rate=1e-3)
	autoencoder.compile(loss="mse", optimizer=opt)

	EPOCHS = 10
	BS = 200

	t0 = time.time()
	H = autoencoder.fit(
		trainXround, trainXround,
		#validation_data=(testX, testX),
		epochs=EPOCHS,
		batch_size=BS)

	print('time elapsed: ', time.time() - t0)

	autoencoder.save('tsie_2D_CAE_gs_p1round')


if __name__ == '__main__':
	main()