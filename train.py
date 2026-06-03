import os
import numpy as np
np.Inf = np.inf  # Patch for Keras/Numpy 2.0 compatibility
import tensorflow as tf
from data_generator import RockGlacierDataGenerator
from models.swin_unet import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FEAT = "dataset/train/features"
TRAIN_MASK = "dataset/train/masks"
VAL_FEAT   = "dataset/val/features"
VAL_MASK   = "dataset/val/masks"
CKPT_DIR   = "checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (baseline 128x128 full-image model)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 1e-4
INPUT_SHAPE = (128, 128, 12)

# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────
print("Initializing Data Generators...")
train_gen = RockGlacierDataGenerator(TRAIN_FEAT, TRAIN_MASK, batch_size=BATCH_SIZE,
                                      target_size=INPUT_SHAPE[:2], augment=True)
val_gen   = RockGlacierDataGenerator(VAL_FEAT, VAL_MASK, batch_size=BATCH_SIZE,
                                      target_size=INPUT_SHAPE[:2], augment=False, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Loss: weighted BCE + Dice  (baseline)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (all at threshold 0.5; logged every epoch by CSVLogger)
# ─────────────────────────────────────────────────────────────────────────────
def dice_coef(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    yt = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(yt * y_pred_bin)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(yt) + tf.reduce_sum(y_pred_bin) + 1e-6)


def precision_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(yt * yp); fp = tf.reduce_sum(yp * (1.0 - yt))
    return (tp + 1e-6) / (tp + fp + 1e-6)


def recall_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(yt * yp); fn = tf.reduce_sum(yt * (1.0 - yp))
    return (tp + 1e-6) / (tp + fn + 1e-6)


def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred); r = recall_m(y_true, y_pred)
    return (2.0 * p * r + 1e-6) / (p + r + 1e-6)


def iou_m(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32); yt = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter
    return (inter + 1e-6) / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Build Model
# ─────────────────────────────────────────────────────────────────────────────
print("Building Model...")
model = build_model(input_shape=INPUT_SHAPE, num_classes=1, num_heads=8)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=combined_loss,
              metrics=['accuracy', dice_coef, precision_m, recall_m, f1_m, iou_m])

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(os.path.join(CKPT_DIR, 'best_model.weights.h5'),
                    monitor='val_dice_coef', mode='max',
                    save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(os.path.join(CKPT_DIR, 'training_history.csv')),
]

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
print("Starting Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)
print("Training Complete.")
