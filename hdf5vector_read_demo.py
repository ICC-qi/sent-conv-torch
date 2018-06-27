import h5py
myFile = h5py.File('/home/icc-qi/sent-conv-torch-master/H04featurevector.hdf5', 'r')

# The '...' means retrieve the whole tensor
myFile.keys()
data = myFile['H04C1I1'][0]
print(data)
myFile.close()
