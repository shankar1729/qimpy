import h5py
from mpi4py import MPI
import numpy as np
import torch
import qimpy as qp

class HDF5_io:
    '''
    Handles HDF5 file i/o for checkpoint file.
    '''

    def __init__(self, filename: str):

        self.filename = filename
        self.f = h5py.File(self.filename, 'a', driver='mpio', comm=MPI.COMM_WORLD)
        # Will read/write if file exists, or create otherwise

    def create_dataset(self, header: str, shape: tuple):
        '''
        Creates a datasdet inside the hdf5 file
        Inputs:
         - header: the hdf5 dataset header (path).
                   This should mirror the yaml "path" if yaml path exists
         - shape:  tuple of the number of elements in each dataset dimension.
        Outputs:
          none
        '''

        # TODO: should there be a note given if dataset already exists?
        if header in self.f: # dataset exists already
            return
        else:
            self.f.create_dataset(header, shape)

    def add_dataset_attribute(self, header: str, attribute_key: str,
                              attribute_value: str):
        '''
        Adds an attribute to an existing hdf5 dataset
        Inputs:
         - header: hdf5 dataset header
         - attribute_key: attribute name to be attached to hdf5 dataset
         - attribute_value: attribute description
        Outputs:
         none
        '''

        dset = self.f[header]
        dset.attrs[attribute_key] = attribute_value

    def write_to_dataset(self, header: str, data: torch.Tensor, offset: tuple):
        '''
        Writes data to an existing hdf5 dataset
        the dataset can be of arbitrary dimension, but must match the
        dimensions specified on creation of the hdf5 dataset
        (see create_dataset). Note that this writes ALL of data
        to the dataset, which will be a subset of the entire dataset, as
        determined by the mpi parallelization scheme. It is the calling
        object's responsibility to set the offsets appropriately.
        Inputs:
         - header: dataset header/ path inside hdf5 file to dataset
         - data: local data on mpi task to be added to the dataset.
         - offset: where to start writing the local data inside the dataset
        Outputs:
         none
        '''

        dset = self.f[header]
        # need to make sure it is on the cpu instead of gpu
        assert len(offset) == len(data.shape)
        #s_i is the size of the shape dimension indexed by i
        index = tuple(slice(offset[i], offset[i] + s_i) for i, s_i in enumerate(data.shape))

        dset[index] = data

    def read_dataset(self, header: str, offset: tuple, size: tuple):
        '''
        Reads a portion (or all) of a dataset into the program.
        Inputs:
         - header: dataset header/ path inside hdf5 file to dataset
         - offset: where to start reading the data inside the dataset
         - size: how much data is being read in each dimension
        Outputs:
         - data: torch tensor of data read from the dataset.
        Example:
            A 2-dimension dataset of size (100, 100) is located at path header
            inside the hdf5 file. MPI task 0 needs all columns of data for
            the first two rows. So, for MPI task 0, offset = (0, 0)
            and size = (2, 100). If MPI task 1 needs the next two rows,
            offset = (2, 0) and size = (2, 100)
        '''

        dset = self.f[header]
        assert len(dset.shape) == len(offset)

        index = tuple(slice(offset[i], offset[i] + size[i]) for i, s_i in enumerate(dset.shape))
        data = torch.from_numpy(dset[index])
        return data

    def close_file(self):

        self.f.flush() # may be unnecessary
        self.f.close()

    def open_file(self):

        # We can add optional argument to set the driver if necessary
        self.f = h5py.File(self.filename, 'a', driver='mpio', comm=MPI.COMM_WORLD)
