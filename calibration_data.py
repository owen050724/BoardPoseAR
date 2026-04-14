"""Default camera calibration constants for BoardPoseAR.

These values are copied from LensRectify's calibration result so the default
pose demo can run without loading a .npz/.json calibration artifact.
"""

CAMERA_MATRIX = [
    [1424.250250171535, 0.0, 671.921287623599],
    [0.0, 1423.805542559728, 338.440846323529],
    [0.0, 0.0, 1.0],
]

DIST_COEFFS = [[0.179046439643, 0.286167456181, 0.000297618823, -0.000910108931, -4.124906944596]]

PATTERN_SIZE = (10, 7)
SQUARE_SIZE = 1.0
# Width, height of the images used for this calibration.
IMAGE_SIZE = (1280, 720)
