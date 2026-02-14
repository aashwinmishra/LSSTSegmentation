from astropy.io import fits
import numpy as np
import torch


def load_lsst_fits(filepath: str)->np.array:
    """
    Utility function to load LSST data locally, without the RSS Butler.
    Assumes that the FITS file has arragement order: 
    [header, image, mask, variance].
    Args:
      filepath: absolute address of the FITS file.
    Returns:
      Image as a numpy array.
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data.astype(np.float32)
    return data
