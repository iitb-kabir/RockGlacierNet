"""
RockGlacierDataGenerator — loads full variable-size GeoTIFF feature/mask pairs
from dataset/{split}/features|masks, normalizes, and pads/crops to a fixed
target_size for the 128x128 baseline model.

This is the original full-image pipeline (restored). Images smaller than
target_size are zero-padded bottom/right; larger images are center-cropped
(random-cropped when augment=True).
"""

import os
import glob
import numpy as np
import rasterio
import tensorflow as tf
import math


# Global means and standard deviations computed during dataset validation
GLOBAL_MEAN = np.array([
    0.324, 0.335, 0.344, 0.357, 0.251, 0.229,
    0.038, -0.061, 0.085, 5126.0, 17.4, 189.3
], dtype=np.float32).reshape(1, 1, 12)

GLOBAL_STD = np.array([
    0.183, 0.169, 0.160, 0.134, 0.058, 0.050,
    0.075, 0.096, 0.301, 282.6, 9.9, 110.8
], dtype=np.float32).reshape(1, 1, 12)


class RockGlacierDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features_dir, masks_dir, batch_size=8, target_size=(128, 128),
                 shuffle=True, augment=False):
        self.features_dir = features_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment

        self.feature_files = sorted(glob.glob(os.path.join(self.features_dir, 'feature_*.tif')))
        self.mask_files = sorted(glob.glob(os.path.join(self.masks_dir, 'mask_*.tif')))

        assert len(self.feature_files) == len(self.mask_files), "Mismatch between feature and mask counts."
        assert len(self.feature_files) > 0, f"No feature_*.tif found in {self.features_dir}"

        self.indices = np.arange(len(self.feature_files))

        self.GLOBAL_MEAN = GLOBAL_MEAN
        self.GLOBAL_STD = GLOBAL_STD

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.feature_files) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        Y_batch = []

        for i in batch_indices:
            f_path = self.feature_files[i]
            m_path = self.mask_files[i]

            with rasterio.open(f_path) as f_src:
                feature = f_src.read()  # (12, H, W)
                feature = np.moveaxis(feature, 0, -1)  # (H, W, 12)

            with rasterio.open(m_path) as m_src:
                mask = m_src.read(1)  # (H, W)
                mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)

            # Replace NoData (-9999) with 0 for safe math
            feature = np.where((feature == -9999) | ~np.isfinite(feature), 0, feature)

            # Normalization
            feature = (feature - self.GLOBAL_MEAN) / (self.GLOBAL_STD + 1e-7)

            # Padding to target_size if smaller
            h, w = feature.shape[:2]
            th, tw = self.target_size

            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)

            if pad_h > 0 or pad_w > 0:
                # pad right and bottom
                feature = np.pad(feature, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

            # Cropping if larger
            h, w = feature.shape[:2]
            if h > th or w > tw:
                if self.augment:
                    # Random crop
                    h_start = np.random.randint(0, h - th + 1) if h > th else 0
                    w_start = np.random.randint(0, w - tw + 1) if w > tw else 0
                else:
                    # Center crop
                    h_start = (h - th) // 2 if h > th else 0
                    w_start = (w - tw) // 2 if w > tw else 0

                feature = feature[h_start:h_start + th, w_start:w_start + tw, :]
                mask = mask[h_start:h_start + th, w_start:w_start + tw, :]

            # Augmentations (Geometric only)
            if self.augment:
                # Random Horizontal Flip
                if np.random.rand() > 0.5:
                    feature = np.fliplr(feature)
                    mask = np.fliplr(mask)
                # Random Vertical Flip
                if np.random.rand() > 0.5:
                    feature = np.flipud(feature)
                    mask = np.flipud(mask)
                # Random Rotations (90, 180, 270)
                k = np.random.randint(0, 4)
                if k > 0:
                    feature = np.rot90(feature, k=k)
                    mask = np.rot90(mask, k=k)

            X_batch.append(feature)
            Y_batch.append(mask)

        return np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
