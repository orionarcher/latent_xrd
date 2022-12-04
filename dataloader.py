"""This file manages data loading for the XRD data.

The XRD_DATA_PATH and DataLoader parameters are meant to be tweaked to match
your environment's settings. You can import xrd_dataloader as your PyTorch
DataLoader.
"""


import h5py
import numpy as np
from os import path
import scipy

import torch
from torch.utils.data import Dataset, DataLoader


XRD_DATA_PATH = "/pscratch/sd/h/hasitha/xrd/icsd_data_189476_10000_cry_extinction_space_density_vol.h5"

if not path.exists(XRD_DATA_PATH):
    raise FileNotFoundError(
        f'File "{XRD_DATA_PATH}" does not exist! '
        "Please set XRD_DATA_PATH in dataloader.py to path of HDF5 file "
        "containing XRD data."
    )

###############################################################################
# DataLoader parameters – These are yours to tweak. Feel free to modify them! #
###############################################################################

BATCH_SIZE = 100
SHUFFLE = True
NUM_WORKERS = 0

###############################################################################


class XRDDataset(Dataset):
    """Represents an XRD dataset.
    
    XRD Dataset must be HDF5.

    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        sample = self.xrd[index][:-5]

        # Add an axis to the array to fit the Perceiver input
        return torch.from_numpy(sample)[None, :]

xrd_dataset = XRDDataset(XRD_DATA_PATH)
xrd_dataloader = DataLoader(
    xrd_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)

class BinaryDataset(Dataset):
    """Represents an XRD dataset.

    XRD Dataset must be HDF5.

    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")
        self.indexes = np.arange(0, 99)

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        zeros = np.random.choice(self.indexes, size=50)
        ones = np.ones(100)
        ones[zeros] = 0


        # Add an axis to the array to fit the Perceiver input
        return torch.from_numpy(ones)[None, :]


binary_dataset = BinaryDataset(XRD_DATA_PATH)
binary_dataloader = DataLoader(
    binary_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)


class SquareXRDDataset(Dataset):
    """Represents an XRD dataset.

    XRD Dataset must be HDF5.
    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        sample = self.xrd[index][:-5]
        edge_sqrt = 10
        tile = np.reshape(np.arange(edge_sqrt ** 2), (edge_sqrt,edge_sqrt))
        a = np.repeat(tile, edge_sqrt, axis=0)
        b = np.repeat(a, edge_sqrt, axis=1)
        tiled = np.tile(tile, (edge_sqrt, edge_sqrt))
        arr = tiled + b * edge_sqrt ** 2
        sample = sample[arr]

        square = sample.reshape(1, 100, 100)

        # Add an axis to the array to fit the Perceiver input
        return torch.from_numpy(square).float()

square_xrd_dataset = SquareXRDDataset(XRD_DATA_PATH)
square_xrd_dataloader = DataLoader(
    square_xrd_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)

class SquareXRDDatasetGaussian(Dataset):
    """Represents an XRD dataset.

    XRD Dataset must be HDF5.
    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        sample = self.xrd[index][:-5]
        blured = scipy.ndimage.gaussian_filter1d(sample, 4)
        square = blured.reshape(1, 100, 100)
        

        out = torch.from_numpy(square).float()

        # Add an axis to the array to fit the Perceiver input
        return out


square_xrd_dataset_gaussian = SquareXRDDatasetGaussian(XRD_DATA_PATH)
square_xrd_dataloader_gaussian = DataLoader(
    square_xrd_dataset_gaussian,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)

class SquareBinaryDataset(Dataset):
    """Represents an XRD dataset.
    XRD Dataset must be HDF5.
    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")
        self.indexes = np.arange(0, 99)

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        zeros = np.random.choice(self.indexes, size=100)
        ones = np.ones(10000)
        ones[zeros] = 0

        # Add an axis to the array to fit the Perceiver input
        return torch.from_numpy(ones.reshape(1, 100, 100)).float()


square_binary_dataset = SquareBinaryDataset(XRD_DATA_PATH)
square_binary_dataloader = DataLoader(
    square_binary_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)

class SquareBinaryDatasetGaussian(Dataset):
    """Represents an XRD dataset.
    XRD Dataset must be HDF5.
    file_path: Path to HDF5 (.h5) file.
    """

    def __init__(self, file_path: str):
        file = h5py.File(file_path, "r")
        self.indexes = np.arange(0, 10000)

        self.xrd: h5py.Dataset = file["data"]
        self.num_samples: int = self.xrd.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns an XRD spectra array with shape (10005,)."""

        # Last 5 elements in XRD data aren't part of the spectra
        ones = np.random.choice(self.indexes, size=100)
        zeros = np.zeros(10000)
        zeros[ones] = 1
        
        blured = scipy.ndimage.gaussian_filter1d(zeros, 4)
        square = blured.reshape(1, 100, 100)

        # Add an axis to the array to fit the Perceiver input
        return torch.from_numpy(square).float()


square_binary_dataset_gaussian = SquareBinaryDatasetGaussian(XRD_DATA_PATH)
square_binary_dataloader_gaussian = DataLoader(
    square_binary_dataset_gaussian,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)