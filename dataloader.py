"""This file manages data loading for the XRD data.

The XRD_DATA_PATH and DataLoader parameters are meant to be tweaked to match
your environment's settings. You can import xrd_dataloader as your PyTorch
DataLoader.
"""


import h5py
import numpy as np
from os import path
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

BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 0

###############################################################################


class XRDDataset(Dataset):
    """Represents an XRD dataset.
    
    XRD Dataset must be HDF5. See
    https://drive.google.com/file/d/1lWCgPNMRAjxH9pqdOKU7hfCNVVKjEcWp/view
    for example.

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
        return self.xrd[index]


xrd_dataset = XRDDataset(XRD_DATA_PATH)
xrd_dataloader = DataLoader(
    xrd_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)
