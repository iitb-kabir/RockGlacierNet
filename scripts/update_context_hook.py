#!/usr/bin/env python3
"""
Stop-hook context updater for RockGlacierNet.

Runs as a plain shell command — no Claude Code permission system involved.
Appends a timestamped session-end entry to agent.md and updates
project_state.md with current git + training state.

Called by .claude/settings.json Stop hook.
"""

import os
import sys
import subprocess
import glob
from datetime import datetime, timezone

ROOT    = "/data1/nasiruddink/rockglacier/RockGlacierNet"
AGENT   = os.path.join(ROOT, "agent.md")
MEMORY  = "/home/nasiruddink/.claude/projects/-data1-nasiruddink-rockglacier-RockGlacierNet/memory"
STATE   = os.path.join(MEMORY, "project_state.md")
MEMIDX  = os.path.join(MEMORY, "MEMORY.md")


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=ROOT,
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return ""


def git_state():
    log     = run("git log --oneline -5")
    status  = run("git status --short")
    return log, status


def training_state():
    csv = os.path.join(ROOT, "outputs/checkpoints/training_history.csv")
    ckpt = os.path.join(ROOT, "outputs/checkpoints/best_model.weights.h5")
    has_ckpt = os.path.exists(ckpt)
    if not os.path.exists(csv):
        return None, has_ckpt
    try:
        # Read last row without pandas dependency
        with open(csv) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return None, has_ckpt
        header = lines[0].split(",")
        last   = lines[-1].split(",")
        row    = dict(zip(header, last))
        return row, has_ckpt
    except Exception:
        return None, has_ckpt


def patch_counts():
    counts = {}
    for split in ("train", "val", "test"):
        d = os.path.join(ROOT, "patches", split, "features")
        if os.path.isdir(d):
            counts[split] = len(glob.glob(os.path.join(d, "*.npy")))
        else:
            counts[split] = 0
    return counts


def append_agent_md(now_str, log, status, row, has_ckpt, patches):
    lines = [
        "",
        "---",
        "",
        f"## Session — {now_str}",
        "",
        "*(Auto-recorded by Stop hook — run `/update-context` for full intelligent summary)*",
        "",
    ]

    lines += ["**Git state:**"]
    if log:
        for l in log.splitlines():
            lines.append(f"- {l}")
    else:
        lines.append("- (no commits)")
    if status:
        lines.append(f"- Modified: {', '.join(status.splitlines()[:5])}")
    lines.append("")

    lines += ["**Training state:**"]
    lines.append(f"- Checkpoint: {'exists' if has_ckpt else 'NOT FOUND'} (best_model.weights.h5)")
    if row:
        epoch   = row.get("epoch", "?")
        iou     = row.get("val_iou_m",     row.get("iou_m", "?"))
        dice    = row.get("val_dice_coef", "?")
        loss    = row.get("val_loss",      row.get("loss", "?"))
        prec    = row.get("val_precision_m", "?")
        rec     = row.get("val_recall_m",    "?")
        lines.append(f"- Last epoch: {epoch}  |  val_iou={iou}  val_dice={dice}  val_loss={loss}")
        lines.append(f"- val_precision={prec}  val_recall={rec}")
    else:
        lines.append("- No training history found")
    lines.append("")

    if any(patches.values()):
        lines += ["**Patches:**"]
        for sp, n in patches.items():
            lines.append(f"- {sp}: {n} patches")
        lines.append("")

    lines.append("**Next steps:** *(fill in manually or run /update-context)*")
    lines.append("")

    entry = "\n".join(lines)

    os.makedirs(os.path.dirname(AGENT) if os.path.dirname(AGENT) else ".", exist_ok=True)
    with open(AGENT, "a") as f:
        f.write(entry)


def update_memory(now_str, row, has_ckpt, log):
    os.makedirs(MEMORY, exist_ok=True)

    # --- MEMORY.md index ---
    mem_content = f"""# RockGlacierNet Memory Index

**Last Updated**: {now_str}

- [project_state.md](project_state.md) — Full pipeline, model architecture, dataset stats, training metrics
"""
    with open(MEMIDX, "w") as f:
        f.write(mem_content)

    # --- project_state.md ---
    iou  = dice = loss = epoch = "unknown"
    if row:
        epoch = row.get("epoch", "?")
        iou   = row.get("val_iou_m",     row.get("iou_m", "?"))
        dice  = row.get("val_dice_coef", "?")
        loss  = row.get("val_loss",      row.get("loss", "?"))

    last_commit = log.splitlines()[0] if log else "unknown"

    state_content = f"""---
name: project-state
description: Full pipeline, model architecture, dataset stats, and current training results for RockGlacierNet
metadata:
  type: project
  lastUpdated: "{now_str}"
---

## What this project does
Binary semantic segmentation of rock glaciers in Sikkim Himalayas.
Input: 12-band GeoTIFFs (Sentinel-2 + DEM). Output: binary glacier/background mask.

## Current Training State (auto-updated {now_str})
- Last commit: {last_commit}
- Checkpoint: {'EXISTS' if has_ckpt else 'MISSING'} — outputs/checkpoints/best_model.weights.h5
- Last epoch: {epoch}
- val_iou_m: {iou}  |  val_dice_coef: {dice}  |  val_loss: {loss}

## Baseline (old 128×128 model, 70 epochs)
- val_iou=0.5825, val_dice=0.7618, precision=0.6233, recall=0.8990, f1=0.7362

## Current Result (new 64×64 model, 58 epochs — REGRESSION)
- val_iou=0.3308 — IoU dropped 44% vs baseline
- Root cause: 64px patches too small for spatial context (glaciers are 40–60px wide)
- Fix needed: increase patch_size to 128 or use full-image variable-size training

## Active Architecture
- Model: ResidualSwinUNET_64x64 (models/swin_unet.py)
- Input: (64, 64, 12) — Output: (64, 64, 1) sigmoid
- Encoder: 3 pool layers, filters 32→64→128→256
- Bottleneck: 8×8×256 + 2× Swin transformer blocks (64 tokens)
- Loss: Focal(α=0.75,γ=2.0)×2 + Dice | Optimizer: Adam lr=1e-4

## Pipeline Run Order
1. python preprocess/extract_patches.py
2. python preprocess/validate_alignment.py
3. python train.py
4. python evaluate.py

## Key Files
- preprocess/patch_config.py    — patch_size=64, stride=32, bg_keep=0.25
- data_generator.py             — loads .npy patches, z-score normalizes
- outputs/checkpoints/training_history.csv — per-epoch metrics
- agent.md                      — full session log and project index
"""
    with open(STATE, "w") as f:
        f.write(state_content)


def main():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    log, status       = git_state()
    row, has_ckpt     = training_state()
    patches           = patch_counts()

    append_agent_md(now_str, log, status, row, has_ckpt, patches)
    update_memory(now_str, row, has_ckpt, log)

    print(f"Context updated — agent.md entry appended ({now_str}).")


if __name__ == "__main__":
    main()
