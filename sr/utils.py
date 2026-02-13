import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import lsst.geom as geom #yes, you should run it on the LSST server.


def simple_norm(image_array: np.array, 
                device: str='cpu', 
                max_dim: int=1024)->tuple:
    """
    Robust normalization for astronomical data.
    Args:
        image_array: Raw pixel data.
        device: Device of the returned tensor.
        max_dim: Maximum edge size for the output (resizes if larger).
    Returns:
        tuple with (tensor_BCHW, image_uint8, scale_factor)
    """
    image_array = np.nan_to_num(image_array, nan=0.0)
    vmin, vmax = np.percentile(image_array, [1, 99.5])
    norm = np.clip((image_array - vmin) / (vmax - vmin), 0, 1) * 255.0
    h, w = norm.shape
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(norm.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = norm.astype(np.uint8)
        scale = 1.0
    tensor = torch.from_numpy(small/255.).float()[None, None].to(device)
    return tensor, small, scale


def crop_to_sky_region(calexp, 
                       sky_corners: list)-> np.array:
    """
    Crops an LSST exposure (calexp) to a specific sky polygon. Ensures that two different visits cover 
    the exact same physical area.
    Args:
        calexp (lsst.afw.image.ExposureF): The LSST data object.
        sky_corners: List of lsst.geom.SpherePoint (corners of the Coadd patch).
    Returns:
        np.array corresponndinng to Cropped pixel array (None if no overlap).
    """
    wcs = calexp.getWcs()
    arr = calexp.image.array
    h, w = arr.shape
    pixel_points = [wcs.skyToPixel(sky) for sky in sky_corners]
    xs = [p.getX() for p in pixel_points]
    ys = [p.getY() for p in pixel_points]
    min_x = max(0, int(min(xs)) - 50) #TODO: Test if 50 is optimal!
    max_x = min(w, int(max(xs)) + 50)
    min_y = max(0, int(min(ys)) - 50)
    max_y = min(h, int(max(ys)) + 50)
    if min_x >= max_x or min_y >= max_y:
        return None 
    crop = arr[min_y:max_y, min_x:max_x]
    if crop.shape[0] < 100 or crop.shape[1] < 100:
        return None
    return crop


def make_rgb_proof(img_A, 
                   img_B_warped):
    """
    Creates an RGB composite to verify alignment.
    Red = Image A, Green = Image B. Yellow stars indicate perfect alignment.
    """
    h, w = img_A.shape
    proof = np.zeros((h, w, 3), dtype=np.uint8)
    proof[:, :, 0] = img_A           
    proof[:, :, 1] = img_B_warped    
    return proof

