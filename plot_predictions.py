import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_generator import RockGlacierDataGenerator
from models.swin_unet import build_model

# Patch for Keras/Numpy 2.0 compatibility
np.Inf = np.inf

TEST_FEAT = "dataset/test/features"
TEST_MASK = "dataset/test/masks"
CKPT_PATH = "checkpoints/best_model.weights.h5"
OUTPUT_DIR = "reuslt/comparisons"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Loss for model loading
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_f, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_f, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    weighted_bce = bce * 2.0
    dl = dice_loss(y_true, y_pred)
    return weighted_bce + dl

print("Building Model and Loading Weights...")
model = build_model(input_shape=(128, 128, 12), num_classes=1, num_heads=8)
model.compile(optimizer='adam', loss=combined_loss)
model.load_weights(CKPT_PATH)

print("Initializing Test Data Generator...")
test_gen = RockGlacierDataGenerator(TEST_FEAT, TEST_MASK, batch_size=1, augment=False, shuffle=False)

print("Generating Comparison Plots...")
# Generate plots for all test examples
num_samples = len(test_gen)

for i in range(num_samples):
    x, y_true = test_gen[i]
    y_pred = model.predict(x, verbose=0)
    
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)[0, :, :, 0]
    y_true_bin = y_true[0, :, :, 0]
    
    # Extract RGB bands for visualization (Assuming Band 3=Red, Band 2=Green, Band 1=Blue)
    # The generator normalizes the features. We'll just grab the un-normalized version 
    # to plot it properly, or just use the normalized ones and stretch them.
    # We will use the normalized features and clip them for display.
    # Note: Band order in data_generator: 0=Blue, 1=Green, 2=Red
    r = x[0, :, :, 2]
    g = x[0, :, :, 1]
    b = x[0, :, :, 0]
    
    rgb = np.dstack((r, g, b))
    
    # Min-max stretch for display
    rgb_min, rgb_max = np.percentile(rgb, (2, 98))
    if rgb_max > rgb_min:
        rgb_disp = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
    else:
        rgb_disp = np.zeros_like(rgb)
        
    file_path = test_gen.feature_files[i]
    base_name = os.path.basename(file_path).replace('feature_', '').replace('.tif', '')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_disp)
    axes[0].set_title(f"Optical (RGB) - {base_name}")
    axes[0].axis('off')
    
    axes[1].imshow(y_true_bin, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(y_pred_bin, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"compare_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

print(f"Successfully saved {num_samples} comparison plots in '{OUTPUT_DIR}'")
