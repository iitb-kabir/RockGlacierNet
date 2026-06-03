"""
Attention U-Net for RockGlacierNet.

Architecture: Oktay et al. (2018) "Attention U-Net: Learning Where to Look
for the Pancreas" — standard U-Net encoder/decoder with soft attention gates
on every skip connection.

Input : (128, 128, 12)
Output: (128, 128,  1) sigmoid — binary glacier mask

Encoder filters : 64 → 128 → 256 → 512
Bottleneck      : 512
Decoder filters : 256 → 128 → 64 → 32
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    BatchNormalization, Activation, concatenate, Multiply
)
from tensorflow.keras.models import Model


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _double_conv(x, filters, dropout=0.0, name=""):
    """Two Conv3×3 + BN + ReLU blocks (standard U-Net block)."""
    x = Conv2D(filters, 3, padding='same', name=f'{name}_c1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_r1')(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f'{name}_dp')(x)
    x = Conv2D(filters, 3, padding='same', name=f'{name}_c2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Activation('relu', name=f'{name}_r2')(x)
    return x


def _attention_gate(x, g, inter_filters, name=""):
    """
    Soft attention gate.

    Parameters
    ----------
    x             : skip-connection features  (H, W, C_x)  — from encoder
    g             : gating signal             (H, W, C_g)  — from decoder (same spatial size)
    inter_filters : intermediate channel size (typically C_x // 2)

    Returns
    -------
    Attention-weighted skip features  (H, W, C_x)
    """
    # Project skip and gating to inter_filters
    theta = Conv2D(inter_filters, 1, padding='same',
                   use_bias=False, name=f'{name}_theta')(x)
    phi   = Conv2D(inter_filters, 1, padding='same',
                   use_bias=False, name=f'{name}_phi')(g)

    # Add → ReLU → 1×1 conv → sigmoid  =  attention coefficient map
    f     = Activation('relu',    name=f'{name}_relu')(theta + phi)
    psi   = Conv2D(1, 1, padding='same',
                   use_bias=True, name=f'{name}_psi')(f)
    alpha = Activation('sigmoid', name=f'{name}_sigmoid')(psi)

    # Scale skip features by attention map (broadcast over channels)
    return Multiply(name=f'{name}_out')([x, alpha])


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

def build_attention_unet(input_shape=(128, 128, 12), num_classes=1):
    """
    Build Attention U-Net.

    Parameters
    ----------
    input_shape : tuple  default (128, 128, 12)
    num_classes : int    1 for binary segmentation

    Returns
    -------
    tf.keras.Model
    """
    inputs = Input(input_shape, name='input')

    # ── Encoder ──────────────────────────────────────────────────
    # Each encoder level: double_conv → save skip → pool
    e1 = _double_conv(inputs, 64,  dropout=0.1, name='enc1')   # 128×128, 64ch
    p1 = MaxPooling2D(2, name='pool1')(e1)

    e2 = _double_conv(p1,     128, dropout=0.1, name='enc2')   # 64×64, 128ch
    p2 = MaxPooling2D(2, name='pool2')(e2)

    e3 = _double_conv(p2,     256, dropout=0.2, name='enc3')   # 32×32, 256ch
    p3 = MaxPooling2D(2, name='pool3')(e3)

    e4 = _double_conv(p3,     512, dropout=0.2, name='enc4')   # 16×16, 512ch
    p4 = MaxPooling2D(2, name='pool4')(e4)

    # ── Bottleneck ───────────────────────────────────────────────
    b  = _double_conv(p4,     512, dropout=0.3, name='btn')    # 8×8,  512ch

    # ── Decoder ──────────────────────────────────────────────────
    # Each decoder level:
    #   1. upsample bottleneck / previous decoder feature
    #   2. attention_gate(skip=encoder_skip, g=upsampled)
    #   3. concatenate(attended_skip, upsampled)
    #   4. double_conv

    # -- Level 4 --
    u4  = Conv2DTranspose(256, 2, strides=2, padding='same', name='up4')(b)
    ag4 = _attention_gate(x=e4, g=u4, inter_filters=256, name='ag4')  # e4:512, u4:256
    d4  = _double_conv(concatenate([ag4, u4], name='cat4'),
                       256, dropout=0.2, name='dec4')                  # 16×16, 256ch

    # -- Level 3 --
    u3  = Conv2DTranspose(128, 2, strides=2, padding='same', name='up3')(d4)
    ag3 = _attention_gate(x=e3, g=u3, inter_filters=128, name='ag3')  # e3:256, u3:128
    d3  = _double_conv(concatenate([ag3, u3], name='cat3'),
                       128, dropout=0.2, name='dec3')                  # 32×32, 128ch

    # -- Level 2 --
    u2  = Conv2DTranspose(64, 2, strides=2, padding='same', name='up2')(d3)
    ag2 = _attention_gate(x=e2, g=u2, inter_filters=64,  name='ag2')  # e2:128, u2:64
    d2  = _double_conv(concatenate([ag2, u2], name='cat2'),
                       64,  dropout=0.1, name='dec2')                  # 64×64,  64ch

    # -- Level 1 --
    u1  = Conv2DTranspose(32, 2, strides=2, padding='same', name='up1')(d2)
    ag1 = _attention_gate(x=e1, g=u1, inter_filters=32,  name='ag1')  # e1:64, u1:32
    d1  = _double_conv(concatenate([ag1, u1], name='cat1'),
                       32,  dropout=0.1, name='dec1')                  # 128×128, 32ch

    # ── Output head ──────────────────────────────────────────────
    outputs = Conv2D(num_classes, 1, activation='sigmoid', name='output')(d1)

    return Model(inputs, outputs, name='AttentionUNet')
