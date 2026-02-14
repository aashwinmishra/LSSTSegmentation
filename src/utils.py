from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
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


def load_local_data(filename: str)->tuple:
    """
    Utility function to load LSST Image and WCS data locally, 
    without the RSS Butler.
    """
    with fits.open(filename) as hdul:
        header = hdul[1].header
        data = hdul[1].data.astype(np.float32)
        wcs = WCS(header)
    return data, wcs

