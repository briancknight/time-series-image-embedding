import numpy as np
import h5py

# requires data from Airbus SAS (link in readme)


def main():

	airbus_train_path = '../airbus_data/dftrain.h5'
	airbus_valid_path = '../airbus_data/dfvalid.h5'

	with h5py.File(airbus_train_path, 'r') as f:
	    # train_data = f['dftrain']
		# List all groups
	    print("Keys: %s" % f.keys())
	    a_group_key = list(f.keys())[0]

	    # Get the data
	    labels = list(f[a_group_key])
	    raw_train_data = np.array(((f[a_group_key])[labels[3]]))

	with h5py.File(airbus_valid_path, 'r') as f:
	    # train_data = f['dftrain']
		# List all groups
	    print("Keys: %s" % f.keys())
	    a_group_key = list(f.keys())[0]

	    # Get the data
	    labels = list(f[a_group_key])
	    raw_valid_data = np.array(((f[a_group_key])[labels[3]]))

	train_len = raw_train_data.size
	print('train_len = ', train_len)

	# split into 1 minute intervals

	# points per minute
	pp_min = 120*512

	split_train_data_per_minute = np.reshape(
		raw_train_data, (int(train_len/(120*512)), (120*512)))

	print(np.shape(split_train_data_per_minute))

	valid_len = raw_valid_data.size
	print('valid_len: ', valid_len)
	split_valid_data_per_minute = np.reshape(
		raw_valid_data, (int(valid_len/(120*512)), (120*512)))

	print(np.shape(split_valid_data_per_minute))
	# divide total data pts by 120 then divide by 512 gives a sequence of... images?
	
if __name__ == '__main__':
	main()