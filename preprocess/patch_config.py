"""
Patch extraction configuration for RockGlacierNet.
Edit values here to experiment with different patch sizes, strides,
and filtering without touching any other code.
"""

PATCH_CONFIG = {
    # ── Patch geometry ────────────────────────────────────────────
    "patch_size": 64,           # spatial size of each square patch (px)
    "stride": 32,               # step between patch origins (px); 32 = 50% overlap

    # ── Filtering ─────────────────────────────────────────────────
    "background_keep_ratio": 0.25,      # fraction of pure-background patches to keep
    "minimum_glacier_percentage": 0.0,  # keep all patches that have ANY glacier px

    # ── Padding ───────────────────────────────────────────────────
    "padding_mode": "reflect",   # "reflect" | "constant" (0-fill)

    # ── Paths ─────────────────────────────────────────────────────
    "dataset_dir": "dataset",    # source: dataset/{split}/features|masks
    "output_dir":  "patches",    # dest:   patches/{split}/features|masks

    # ── Other ─────────────────────────────────────────────────────
    "num_bands": 12,
    "splits": ["train", "val", "test"],
    "random_seed": 42,
}

# ── Quick-experiment presets (swap into PATCH_CONFIG to test) ───────
PRESET_64_STRIDE64 = {**PATCH_CONFIG, "stride": 64}
PRESET_96_STRIDE48 = {**PATCH_CONFIG, "patch_size": 96, "stride": 48}
PRESET_64_NOSTRIDE = {**PATCH_CONFIG, "stride": 64}   # non-overlapping
