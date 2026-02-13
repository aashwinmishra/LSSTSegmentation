# Alignment for LSST Data
Automated feature-based registration of large-baseline, dithered and rotated LSST exposures.

## Overview
This repository contains a robust pipeline for aligning Rubin Observatory (LSST) images using a graph neural network for feature matching. While standard WCS alignment is generally reliable, this tool provides an independent, pixel-based alignment method capable of handling: 
1) Large Dithers: Matching visits with less than 20% overlap.
2) Camera Rotations: Automatically detecting and correcting 90°, 180°, and 270° sensor rotations.
3) Varying Conditions: Robustness against cosmic rays, different PSFs, and background noise.

## Results
These results demonstrate the alignment of two distinct visits (Visit 458669 vs 665665) on Tract 4850.

1. The "Yellow Star" Proof (RGB Composite)
Red: Visit A (Reference) | Green: Visit B (Warped) | Yellow: Perfect Overlap
Note: The stars appear yellow, confirming sub-pixel registration accuracy. Separate red/green dots would indicate failure.

2. Residuals (Difference Map)
The geometric alignment is consistent. Residuals (dark spots) are due to PSF differences ("donuts") and interpolation, not misalignment ("dipoles").

## Repository Structure
.
├── notebooks/

│   ├── 00_setup_guide.md  

│   ├── 01_synthetic_test.ipynb 

│   └── 02_lsst_real_data.ipynb 

├── src/

│   └── utils.py                


## Quick Start (Rubin Science Platform)
1. Environment Setup
The RSP does not have SuperGlue pre-installed. You must clone the repository in your working directory.

2. Run the Alignment
Open notebooks/02_lsst_real_data.ipynb. This notebook performs the following steps:

1. Major Visit Selection: Queries the Butler to find two visits that overlap the center of the patch (avoiding "sliver" overlaps).
2. 2.WCS Cropping: Uses the deepCoadd bounding box to crop both visits to the exact same sky coordinates.
3. Robust Normalization: Applies percentile clipping (1st-99.5th) to remove cosmic rays and scale star intensities.
4. Rotation Search: Brute-force tests 0°, 90°, 180°, and 270° orientations to find the correct sensor geometry.
5. Inference: Runs SuperPoint (feature extraction) and SuperGlue (matching).
6. Verification: Outputs the transformation matrix (H) and visualization plots.

## Key Technical Challenges Solved
The "Sliver" Problem:
Raw overlapping visits often only touch the edge of a patch. This pipeline uses bbox.contains() checks to ensure we only process Major visits with significant overlap, preventing SuperGlue from trying to match a square to a thin rectangle.

Dynamic Range & Cosmic Rays:
Standard MinMax normalization fails on raw calexps due to bright cosmic rays. We use Percentile Normalization to suppress outliers.

Camera Rotation:
LSST sensors rotate relative to the sky. SuperGlue is rotation-invariant only up to about 45°. We implemented a 4-step rotation search loop that rotates the tensor representation of the second visit to automatically lock onto the correct quadrant before fine-tuning.

## Dependencies
torch (PyTorch)
opencv-python (cv2)
lsst.daf.butler
matplotlib

lsst.daf.butler (LSST Science Pipelines)
