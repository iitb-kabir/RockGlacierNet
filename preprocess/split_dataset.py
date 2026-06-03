import os
import glob
import shutil
import random

# Set a fixed seed for reproducibility so the split is always the same
random.seed(42)

H_FEATURE_DIR = r"D:\RockGlacier_Project\harmonized_features"
H_MASK_DIR = r"D:\RockGlacier_Project\harmonized_masks"

DATASET_DIR = r"D:\RockGlacier_Project\dataset"

splits = ['train', 'val', 'test']
for s in splits:
    os.makedirs(os.path.join(DATASET_DIR, s, 'features'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, s, 'masks'), exist_ok=True)

feature_files = sorted(glob.glob(os.path.join(H_FEATURE_DIR, 'feature_*.tif')))

# Shuffle files
random.shuffle(feature_files)

# Calculate split indices (70% train, 15% val, 15% test)
total = len(feature_files)
train_end = int(total * 0.70)
val_end = int(total * 0.85)

train_files = feature_files[:train_end]
val_files = feature_files[train_end:val_end]
test_files = feature_files[val_end:]

def copy_files(files, split_name):
    for f_path in files:
        idx = os.path.basename(f_path).split('_')[1].split('.')[0]
        m_path = os.path.join(H_MASK_DIR, f"mask_{idx}.tif")
        
        if os.path.exists(m_path):
            shutil.copy(f_path, os.path.join(DATASET_DIR, split_name, 'features', f"feature_{idx}.tif"))
            shutil.copy(m_path, os.path.join(DATASET_DIR, split_name, 'masks', f"mask_{idx}.tif"))

print(f"Total files: {total}")
print(f"Train set: {len(train_files)} files")
print(f"Validation set: {len(val_files)} files")
print(f"Test set: {len(test_files)} files")
print("Copying files into dataset directories...")

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"Dataset successfully split and organized in {DATASET_DIR}")
