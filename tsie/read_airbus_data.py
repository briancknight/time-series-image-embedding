import numpy as np
import h5py

# requires data from Airbus SAS (link in readme)
airbus_train_path = '../airbus_data/dftrain.h5'
airbus_valid_path = '../airbus_data/dfvalid.h5'

# # open training file as 'f'
# with h5py.File(airbus_train_path, 'r') as f:
#     train_data = f['default']

# print(train_data.shape)

# # open training file as 'f'
# with h5py.File(airbus_train_path, 'r') as f:
#     valid_data = f['default']

# print(valid_data.shape)

myarray = np.fromfile(airbus_train_path, dtype=float)
print(myarray[0:10])