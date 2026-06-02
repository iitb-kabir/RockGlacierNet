import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    BatchNormalization, Activation, concatenate, Add, Reshape,
    Dense, Dropout, LayerNormalization
)
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------------
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def transformer_block(x, num_heads, embed_dim):
    # Multi-head attention
    x1 = LayerNormalization(epsilon=1e-6)(x)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, dropout=0.1
    )(x1, x1)
    x2 = Add()([attention_output, x])
    
    # FFN
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=[embed_dim * 2, embed_dim], dropout_rate=0.1)
    out = Add()([x3, x2])
    return out

def swin_block(x, num_heads):
    """
    Simplified Swin-Transformer block adapted for 2D inputs.
    Flattens spatial dimensions -> token sequence, applies self-attention.
    Input: (B, H, W, C)
    Output: (B, H, W, C)
    """
    # Note: Keras layer inputs are tensors, we use tf.shape for dynamic shapes
    # but for Reshape we can rely on standard static shapes if possible,
    # or just let Keras handle it via layers.Reshape.
    
    shape = x.shape
    H, W, C = shape[1], shape[2], shape[3]
    
    if H is None or W is None:
        # Fallback to dynamic shaping if static shape isn't available
        # However, tf.keras.layers.Reshape requires static shape tuples with integers or -1.
        tokens = Reshape((-1, C))(x)
    else:
        tokens = Reshape((H * W, C))(x)
        
    # Transformer block
    tokens = transformer_block(tokens, num_heads, C)
    
    if H is None or W is None:
        # We can't reshape back dynamically easily with layers.Reshape, 
        # so we assume static shapes are provided (which they are: 128x128 -> 8x8 in bottleneck)
        pass # Will fail if shape is fully dynamic
    else:
        x = Reshape((H, W, C))(tokens)
    return x


# ---------------------------------------------------------------------------
# Residual Conv Block
# ---------------------------------------------------------------------------
def conv_block(x, filters, name=""):
    """
    Residual-only block adapted to 2D: 
    Main path [Conv2D-BN-ReLU-Conv2D-BN] + Shortcut [Conv2D-BN]
    """
    # Main path
    h = Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(x)
    h = BatchNormalization(name=f'{name}_bn1')(h)
    h = Activation('relu', name=f'{name}_relu1')(h)
    
    h = Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(h)
    h = BatchNormalization(name=f'{name}_bn2')(h)

    # Residual shortcut — 1x1 projection
    shortcut = Conv2D(filters, 1, padding='same', name=f'{name}_shortcut_conv')(x)
    shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # Add main and shortcut
    out = Add(name=f'{name}_add')([h, shortcut])

    out = Activation('relu', name=f'{name}_final_relu')(out)
    return out


# ---------------------------------------------------------------------------
# Full Residual SwinUNETR (2D)
# ---------------------------------------------------------------------------
def build_model(input_shape=(128, 128, 12), num_classes=1, num_heads=8):
    inputs = Input(input_shape, name='input')

    # ── Encoder ──────────────────────────────────────────────
    c1 = conv_block(inputs, 16, name="enc1")        # 128x128, 16ch
    p1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(c1)

    c2 = conv_block(p1, 32, name="enc2")            # 64x64, 32ch
    p2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(c2)

    c3 = conv_block(p2, 64, name="enc3")            # 32x32, 64ch
    p3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(c3)

    c4 = conv_block(p3, 128, name="enc4")           # 16x16, 128ch
    p4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(c4)

    # ── Bottleneck ───────────────────────────────────────────
    b = conv_block(p4, 256, name="bottleneck")      # 8x8, 256ch
    b = swin_block(b, num_heads=num_heads)
    b = swin_block(b, num_heads=num_heads)

    # ── Decoder ──────────────────────────────────────────────
    u4 = Conv2DTranspose(128, 2, strides=2, padding='same', name="up4")(b)
    u4 = concatenate([u4, c4], name="cat4")
    u4 = conv_block(u4, 128, name="dec4")           # 16x16, 128ch

    u3 = Conv2DTranspose(64, 2, strides=2, padding='same', name="up3")(u4)
    u3 = concatenate([u3, c3], name="cat3")
    u3 = conv_block(u3, 64, name="dec3")            # 32x32, 64ch

    u2 = Conv2DTranspose(32, 2, strides=2, padding='same', name="up2")(u3)
    u2 = concatenate([u2, c2], name="cat2")
    u2 = conv_block(u2, 32, name="dec2")            # 64x64, 32ch

    u1 = Conv2DTranspose(16, 2, strides=2, padding='same', name="up1")(u2)
    u1 = concatenate([u1, c1], name="cat1")
    u1 = conv_block(u1, 16, name="dec1")            # 128x128, 16ch

    # ── Output ───────────────────────────────────────────────
    # We use 1 filter and sigmoid activation for binary classification
    outputs = Conv2D(num_classes, 1, activation='sigmoid', name='output')(u1)

    model = Model(inputs=inputs, outputs=outputs, name='Residual_Only_SwinUNETR_2D')
    return model
