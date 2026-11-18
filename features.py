"""Golden-template feature extraction for the component images.

The idea: every component is photographed in roughly the same way, so a defect
is a *local* departure from a clean reference component. The pipeline is

    1. align   - warp each image onto a reference image (ORB keypoints + RANSAC),
                 which removes the rotation and the small shifts between photos;
    2. template- the "golden template" is the per-pixel median of the aligned
                 images that the inspector marked as OK;
    3. diff    - the absolute difference between an aligned image and the
                 template, blurred a little so single noisy pixels do not count;
    4. features- two numbers taken from that difference map.

Both features are divided by the median of the image's own difference map, so an
image that is simply darker, brighter or noisier does not get a high score.

Used by part_b.ipynb (Sections B.2 / B.3) and part_c+d.ipynb (Sections C / D).
"""

from pathlib import Path

import cv2
import numpy as np

# Intensity scale that every image is mapped to before alignment and comparison.
TARGET_MEAN = 128.0
TARGET_STD = 40.0

# Difference-map settings
BLUR_KERNEL = (5, 5)   # smooth the difference map, so one noisy pixel cannot win
MARGIN = 20            # ignore this border, where warping is unreliable
AREA_FACTOR = 6.0      # a pixel counts as "different" above AREA_FACTOR * median difference

FEATURE_NAMES = ["max_diff_norm", "area_norm"]


def load_grey(path, data_dir=Path(".")):
    """Read an image as a float32 greyscale array."""
    image = cv2.imread(str(Path(data_dir) / path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32)


def standardise(grey):
    """Map an image to a fixed mean and standard deviation.

    This is what makes the exposure differences between photos harmless: after
    this step a dark image and a bright image of the same component look alike.
    """
    return (grey - grey.mean()) / (grey.std() + 1e-6) * TARGET_STD + TARGET_MEAN


class Aligner:
    """Warps images onto one reference image.

    ORB finds keypoints in both images, the matcher pairs them, and RANSAC fits a
    partial affine transform (rotation, translation, uniform scale) from the pairs
    while throwing out the mismatches. If that fails the image is returned
    unchanged and the caller is told, so failures stay visible instead of silently
    producing a huge difference map.
    """

    def __init__(self, reference, n_features=1500, max_matches=80, min_inliers=8):
        self.reference = reference
        self.max_matches = max_matches
        self.min_inliers = min_inliers
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._kp, self._des = self._orb.detectAndCompute(reference.astype(np.uint8), None)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if self._des is None:
            raise ValueError("no keypoints found in the reference image")

    def align(self, image):
        """Return (aligned_image, ok). ok is False when the image could not be aligned."""
        kp, des = self._orb.detectAndCompute(image.astype(np.uint8), None)
        if des is None or len(kp) < self.min_inliers:
            return image, False

        matches = self._matcher.match(des, self._des)
        if len(matches) < self.min_inliers:
            return image, False
        matches = sorted(matches, key=lambda m: m.distance)[:self.max_matches]

        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([self._kp[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return image, False

        aligned = cv2.warpAffine(image, matrix, self.reference.shape[::-1],
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned, True


def align_images(files, reference_file, data_dir=Path("."), aligner=None):
    """Align every file onto the reference.

    Returns (aligned, failures): a dict file -> aligned array, and the list of
    files that could not be aligned (they are kept unchanged in the dict).
    """
    if aligner is None:
        aligner = Aligner(standardise(load_grey(reference_file, data_dir)))
    aligned, failures = {}, []
    for file in files:
        warped, ok = aligner.align(standardise(load_grey(file, data_dir)))
        aligned[file] = warped
        if not ok:
            failures.append(file)
    return aligned, failures


def build_template(aligned_ok):
    """The golden template: the per-pixel median of the aligned OK images.

    The median ignores the odd image with a mark on it, so the template stays a
    picture of a clean component.
    """
    stack = np.stack(list(aligned_ok))
    if len(stack) < 5:
        raise ValueError(f"need at least 5 OK images for a template, got {len(stack)}")
    return np.median(stack, axis=0)


def difference_map(aligned, template, margin=MARGIN):
    """Blurred absolute difference between an aligned image and the template."""
    diff = cv2.GaussianBlur(np.abs(aligned - template), BLUR_KERNEL, 0)
    return diff[margin:-margin, margin:-margin]


def features_from_diff(diff):
    """The two numeric features of one difference map.

    max_diff_norm - the strongest local difference, divided by the median
                    difference of that same image: "how much worse is the worst
                    spot than this image's normal level of disagreement".
    area_norm     - how many pixels differ by more than AREA_FACTOR times that
                    median: the size of the suspicious region.
    """
    median = float(np.median(diff)) + 1e-6
    return {
        "max_diff_norm": float(diff.max() / median),
        "area_norm": int((diff > AREA_FACTOR * median).sum()),
    }


def image_features(path, template, aligner, data_dir=Path(".")):
    """Read one image, align it to the template's reference, return the two features."""
    aligned, ok = aligner.align(standardise(load_grey(path, data_dir)))
    feats = features_from_diff(difference_map(aligned, template))
    feats["aligned"] = ok
    return feats