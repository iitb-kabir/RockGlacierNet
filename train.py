import os
import numpy as np
np.Inf = np.inf  # Patch for Keras/Numpy 2.0 compatibility
import tensorflow as tf
from data_generator import RockGlacierDataGenerator
from models.swin_unet import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

# Set paths
TRAIN_FEAT = r"D:\RockGlacier_Project\dataset\train\features"
TRAIN_MASK = r"D:\RockGlacier_Project\dataset\train\masks"
VAL_FEAT   = r"D:\RockGlacier_Project\dataset\val\features"
VAL_MASK   = r"D:\RockGlacier_Project\dataset\val\masks"
CKPT_DIR   = r"D:\RockGlacier_Project\checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

# Generators
print("Initializing Data Generators...")
train_gen = RockGlacierDataGenerator(TRAIN_FEAT, TRAIN_MASK, batch_size=BATCH_SIZE, augment=True)
val_gen   = RockGlacierDataGenerator(VAL_FEAT, VAL_MASK, batch_size=BATCH_SIZE, augment=False, shuffle=False)

# Loss Function: Combined BCE + Dice
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
    # Apply a weight to BCE for positive class (rock glacier)
    weighted_bce = bce * 2.0
    dl = dice_loss(y_true, y_pred)
    return weighted_bce + dl

# Metrics
def dice_coef(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * y_pred_bin)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(y_pred_bin) + 1e-6)

# Build Model
print("Building Model...")
model = build_model(input_shape=(128, 128, 12), num_classes=1, num_heads=8)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
              loss=combined_loss, 
              metrics=['accuracy', dice_coef])

model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(os.path.join(CKPT_DIR, 'best_model.weights.h5'), 
                    monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(os.path.join(CKPT_DIR, 'training_history.csv'))
]

# Train
print("Starting Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)
print("Training Complete.")
