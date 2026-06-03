import os
import numpy as np
np.Inf = np.inf
import tensorflow as tf
from data_generator import RockGlacierDataGenerator
from models.attention_unet import build_attention_unet
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FEAT  = "dataset/train/features"
TRAIN_MASK  = "dataset/train/masks"
VAL_FEAT    = "dataset/val/features"
VAL_MASK    = "dataset/val/masks"
CKPT_DIR    = "outputs/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 1e-4
INPUT_SHAPE = (128, 128, 12)

# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────
print("Initializing data generators…")
train_gen = RockGlacierDataGenerator(
    TRAIN_FEAT, TRAIN_MASK,
    batch_size=BATCH_SIZE, target_size=INPUT_SHAPE[:2], augment=True)
val_gen = RockGlacierDataGenerator(
    VAL_FEAT, VAL_MASK,
    batch_size=BATCH_SIZE, target_size=INPUT_SHAPE[:2], augment=False, shuffle=False)
print(f"  Train: {len(train_gen.feature_files)} images | Val: {len(val_gen.feature_files)} images")

# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    yt = tf.cast(y_true, tf.float32)
    i  = tf.reduce_sum(yt * y_pred,  axis=[1, 2, 3])
    u  = tf.reduce_sum(yt,           axis=[1, 2, 3]) + \
         tf.reduce_sum(y_pred,       axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * i + smooth) / (u + smooth))


def focal_loss(alpha=0.75, gamma=2.0):
    def _loss(y_true, y_pred):
        yt  = tf.cast(y_true, tf.float32)
        yp  = tf.clip_by_value(y_pred,
                               tf.keras.backend.epsilon(),
                               1.0 - tf.keras.backend.epsilon())
        at  = yt * alpha + (1.0 - yt) * (1.0 - alpha)
        pt  = yt * yp    + (1.0 - yt) * (1.0 - yp)
        return tf.reduce_mean(-at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return _loss


def combined_loss(y_true, y_pred):
    return focal_loss(alpha=0.75, gamma=2.0)(y_true, y_pred) * 2.0 \
           + dice_loss(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (all at t=0.5, logged every epoch)
# ─────────────────────────────────────────────────────────────────────────────
def dice_coef(y_true, y_pred):
    yp = tf.cast(y_pred > 0.5, tf.float32)
    yt = tf.cast(y_true, tf.float32)
    return (2.0 * tf.reduce_sum(yt * yp) + 1e-6) / \
           (tf.reduce_sum(yt) + tf.reduce_sum(yp) + 1e-6)


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
    i  = tf.reduce_sum(yt * yp)
    u  = tf.reduce_sum(yt) + tf.reduce_sum(yp) - i
    return (i + 1e-6) / (u + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Build + compile
# ─────────────────────────────────────────────────────────────────────────────
print("Building Attention U-Net…")
model = build_attention_unet(input_shape=INPUT_SHAPE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=combined_loss,
    metrics=['accuracy', dice_coef, precision_m, recall_m, f1_m, iou_m]
)
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# CSVLogger saves: epoch, loss, accuracy, dice_coef, precision_m, recall_m,
#                  f1_m, iou_m + val_ versions — one row per epoch.
# ─────────────────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(CKPT_DIR, 'attention_unet_best.weights.h5')

callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor='val_iou_m', mode='max',
        save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(
        monitor='val_iou_m', mode='max',
        patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor='val_iou_m', mode='max',
        factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(os.path.join(CKPT_DIR, 'attention_unet_history.csv')),
]

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Attention U-Net…")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)
print("Training complete.")
print(f"Best weights → {ckpt_path}")
print(f"Training log → {CKPT_DIR}/attention_unet_history.csv")
