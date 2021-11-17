import numpy as np
import h5py

# requires data from Airbus SAS (link in readme)


def read_airbus_data():

	airbus_train_path = '../airbus_data/dftrain.h5'
	airbus_valid_path = '../airbus_data/dfvalid.h5'

	with h5py.File(airbus_train_path, 'r') as f:
	    # train_data = f['dftrain']
		# List all groups
	    # print("Keys: %s" % f.keys())
	    a_group_key = list(f.keys())[0]

	    # Get the data
	    labels = list(f[a_group_key])
	    raw_train_data = np.array(((f[a_group_key])[labels[3]]))

	with h5py.File(airbus_valid_path, 'r') as f:
	    # train_data = f['dftrain']
		# List all groups
	    # print("Keys: %s" % f.keys())
	    a_group_key = list(f.keys())[0]

	    # Get the data
	    labels = list(f[a_group_key])
	    raw_valid_data = np.array(((f[a_group_key])[labels[3]]))

	train_len = raw_train_data.size

	# split into 1 minute intervals at 1024 Hz 
	mp_min = 60*1024 # measurments per minute

	split_train_data_per_minute = np.reshape(
		raw_train_data, (int(train_len/(mp_min)), mp_min))

	valid_len = raw_valid_data.size

	split_valid_data_per_minute = np.reshape(
		raw_valid_data, (int(valid_len/mp_min), mp_min))

	return split_train_data_per_minute, split_valid_data_per_minute
