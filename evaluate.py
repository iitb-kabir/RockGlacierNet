import os
import numpy as np
import tensorflow as tf
from data_generator import RockGlacierDataGenerator
from models.swin_unet import build_model
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from PIL import Image

# Patch for Keras/Numpy 2.0 compatibility
np.Inf = np.inf

# Set paths
TEST_FEAT = "dataset/test/features"
TEST_MASK = "dataset/test/masks"
CKPT_PATH = "checkpoints/best_model.weights.h5"
RESULT_DIR = "reuslt"

os.makedirs(RESULT_DIR, exist_ok=True)

# Define Loss and Metrics for loading the model correctly
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

def dice_coef(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * y_pred_bin)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(y_pred_bin) + 1e-6)

# Build Model
print("Building Model...")
model = build_model(input_shape=(128, 128, 12), num_classes=1, num_heads=8)
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', dice_coef])

if not os.path.exists(CKPT_PATH):
    print(f"Error: Checkpoint not found at {CKPT_PATH}")
    print("Please make sure you have trained the model and the weights are saved.")
    exit(1)

print(f"Loading weights from {CKPT_PATH}...")
model.load_weights(CKPT_PATH)

print("Initializing Test Data Generator...")
test_gen = RockGlacierDataGenerator(TEST_FEAT, TEST_MASK, batch_size=1, augment=False, shuffle=False)

print("Starting Evaluation on Test Dataset...")
results = model.evaluate(test_gen, verbose=1)
print("\n--- Initial Evaluation Results ---")
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
print(f"Test Dice Coef: {results[2]:.4f}")
print("----------------------------------\n")

print("Generating predictions and saving masks...")
all_preds = []
all_trues = []

for i in range(len(test_gen)):
    x, y_true = test_gen[i]
    y_pred = model.predict(x, verbose=0)
    
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)
    y_true_bin = y_true.astype(np.uint8)
    
    # Save the prediction as a PNG image
    file_path = test_gen.feature_files[i]
    base_name = os.path.basename(file_path).replace('feature_', 'pred_').replace('.tif', '.png')
    save_path = os.path.join(RESULT_DIR, base_name)
    
    # Extract the 2D mask, scale to 0-255, and save
    pred_img = y_pred_bin[0, :, :, 0] * 255
    img = Image.fromarray(pred_img)
    img.save(save_path)
    
    all_preds.append(y_pred_bin.flatten())
    all_trues.append(y_true_bin.flatten())

print("Calculating overall sklearn metrics...")
y_true_all = np.concatenate(all_trues)
y_pred_all = np.concatenate(all_preds)

precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
iou = jaccard_score(y_true_all, y_pred_all, zero_division=0)

print("\n--- Final Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
print("---------------------")

metrics_path = os.path.join(RESULT_DIR, "evaluation_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("--- Final Evaluation Metrics ---\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"IoU:       {iou:.4f}\n")

print(f"\nAll prediction masks and metrics have been successfully saved to '{RESULT_DIR}'.")
