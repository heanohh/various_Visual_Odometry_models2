# -*- coding: utf-8 -*-


import os
import sys
import json
import random
import re
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# ====================== AUGMENTATION HELPERS ======================
def apply_speckle_noise(frames, prob=0.48, std=0.11):
    """Speckle noise"""
    if random.random() >= prob:
        return frames
    noise = np.random.normal(0, std, frames.shape).astype(np.float32)
    f = frames.astype(np.float32)
    return np.clip(f * (1 + noise), 0, 255).astype(np.uint8)

def apply_brightness_adjustment(frames, prob=0.62, max_delta=0.25):
    """Brightness adjustment"""
    if random.random() >= prob:
        return frames
    delta = random.uniform(-max_delta, max_delta)
    f = frames.astype(np.float32)
    return np.clip(f * (1 + delta), 0, 255).astype(np.uint8)

def apply_time_reversal(frames, labels, prob=0.40):
    """Time reversal (labels도 함께 reverse)"""
    if random.random() >= prob:
        return frames, labels
    return frames[::-1].copy(), labels[::-1].copy()


def apply_frame_skipping(frames, labels, prob=0.33, max_drop_ratio=0.23):
    """Frame skipping + 길이 유지 (labels도 동일하게 skip + last repeat)"""
    if random.random() >= prob:
        return frames, labels
    n = len(frames)
    keep_n = max(int(n * (1 - random.uniform(0.08, max_drop_ratio))), n - 4)
    indices = sorted(random.sample(range(n), keep_n))
    new_frames = frames[indices]
    new_labels = labels[indices]
    if len(new_frames) < n:
        pad = np.repeat(new_frames[-1:], n - len(new_frames), axis=0)
        pad_label = np.repeat(new_labels[-1:], n - len(new_labels), axis=0)
        new_frames = np.concatenate([new_frames, pad], axis=0)
        new_labels = np.concatenate([new_labels, pad_label], axis=0)
    return new_frames, new_labels
# =============================================================

# =========================================
# Sequence builder + data aug
# =========================================
def make_sequences_from_dir_train(dir_path: str, label_names, label_from="end", crop_prob=0.5):
    folder = Path(dir_path)
    if not folder.exists():
        print(f"[skip] folder does not exist: {dir_path}")
        return None, None

    img_paths = list_sorted_images(folder)
    if len(img_paths) == 0:
        print(f"[skip] {dir_path}: found 0 images")
        return None, None

    if len(img_paths) < WINDOW_SIZE:
        print(f"[skip] {dir_path}: not enough frames ({len(img_paths)} < {WINDOW_SIZE})")
        return None, None

    use_frames = min(NUM_FRAMES, len(img_paths))
    img_paths = img_paths[:use_frames]

    n = len(img_paths)

    # ====================== 순차적 연속 블록으로 크롭 적용 구간 선택 ======================
    num_crop_frames = int(n * crop_prob)  # 예: 200 * 0.2 = 40
    if num_crop_frames < 1:
        num_crop_frames = 1

    # 연속된 블록 시작 위치를 랜덤으로 선택 (중간에 띄우지 않음)
    max_start = n - num_crop_frames
    crop_start_idx = random.randint(0, max_start)
    crop_end_idx = crop_start_idx + num_crop_frames  # [crop_start_idx : crop_end_idx)

    print(f"[crop block] {dir_path} → cropping frames {crop_start_idx} to {crop_end_idx - 1} "
          f"({num_crop_frames} frames)")

    frames = []
    crop_box = None

    for idx, ip in enumerate(img_paths):
        try:
            with Image.open(ip) as im:
                im = im.convert("L")
                orig_w, orig_h = im.size
                crop_w, crop_h = RESIZE_HW

                # 선택된 연속 블록 안에 있으면 중앙 크롭 적용
                if crop_start_idx <= idx < crop_end_idx:
                    if crop_box is None:
                        # 원래 코드와 동일한 중앙 크롭
                        left = (orig_w - crop_w) / 2
                        top = (orig_w - crop_w) / 2
                        crop_box = (left, top, left + crop_w, top + crop_h)

                    im = im.crop(crop_box)
                # else: 크롭 없이 진행

                # resize
                im = im.resize(RESIZE_HW, Image.BILINEAR)
                arr = np.array(im, dtype=np.uint8)
                frames.append(arr[None, ...])

        except Exception as e:
            print(f"[error] reading {ip}: {e}")
            return None, None

    frames = np.stack(frames, axis=0)

    # ==================== LABELS ====================
    try:
        labels_all = load_label_excel(folder, EXCEL_NAME, label_names)
    except Exception as e:
        print(f"[error] {dir_path}: {e}")
        return None, None

    F = min(frames.shape[0], labels_all.shape[0])
    frames = frames[:F]
    labels_all = labels_all[:F]

    # ====================== AUGMENTATIONS ======================
    frames = apply_brightness_adjustment(frames, prob=0.2)
    frames, labels_all = apply_time_reversal(frames, labels_all, prob=0.2)
    # ==========================================================

    seqs, ys = [], []
    for s in range(0, F - WINDOW_SIZE + 1, STRIDE):
        e = s + WINDOW_SIZE
        seq = frames[s:e]

        if label_from == "end":
            y = labels_all[e - 1]
        elif label_from == "center":
            y = labels_all[s + WINDOW_SIZE // 2]
        elif label_from == "mean":
            y = labels_all[s:e].mean(axis=0)
        else:
            raise ValueError("label_from must be one of ['end','center','mean']")

        seqs.append(seq)
        ys.append(y)

    if not seqs:
        print(f"[skip] {dir_path}: no windows produced")
        return None, None

    X = np.stack(seqs, axis=0).astype(np.uint8)
    Y = np.stack(ys, axis=0).astype(np.float32)

    print(f"[make_sequences_from_dir_train] {dir_path} → {len(seqs)} sequences | "
          f"continuous crop block: {num_crop_frames} frames (prob≈{crop_prob})")

    return X, Y


def make_sequences_from_dir_valtest(dir_path: str, label_names, label_from="end"):
    folder = Path(dir_path)
    if not folder.exists():
        print(f"[skip] folder does not exist: {dir_path}")
        return None, None

    img_paths = list_sorted_images(folder)
    if len(img_paths) == 0:
        print(f"[skip] {dir_path}: found 0 images")
        return None, None

    # Skip if not enough frames for one sequence
    if len(img_paths) < WINDOW_SIZE:
        print(f"[skip] {dir_path}: not enough frames ({len(img_paths)} < {WINDOW_SIZE})")
        return None, None

    use_frames = min(NUM_FRAMES, len(img_paths)) #
    img_paths = img_paths[:use_frames]

    frames = []
    for ip in img_paths:
        try:
            with Image.open(ip) as im:

                im = im.convert("L").resize(RESIZE_HW, Image.BILINEAR)
                arr = np.array(im, dtype=np.uint8)
                frames.append(arr[None, ...])
        except Exception as e:
            print(f"[error] reading {ip}: {e}")
            return None, None
    frames = np.stack(frames, axis=0)

    try:
        labels_all = load_label_excel(folder, EXCEL_NAME, label_names)
    except Exception as e:
        print(f"[error] {dir_path}: {e}")
        return None, None

    F = min(frames.shape[0], labels_all.shape[0])
    if F < frames.shape[0] or F < labels_all.shape[0]:
        print(f"[warn] {dir_path}: length mismatch -> frames={frames.shape[0]}, labels={labels_all.shape[0]}, using F={F}")
    frames = frames[:F]
    labels_all = labels_all[:F]

    seqs, ys = [], []
    for s in range(0, F - WINDOW_SIZE + 1, STRIDE):
        e = s + WINDOW_SIZE #first try: 20
        seq = frames[s:e]
        #y = labels_all[s:e]

        if label_from == "end": #마지막 데이터만 뽑는건가
            y = labels_all[e - 1]
        elif label_from == "center":
            y = labels_all[s + WINDOW_SIZE // 2]
        elif label_from == "mean":
            y = labels_all[s:e].mean(axis=0)
        else:
            raise ValueError("label_from must be one of ['end','center','mean']")
        seqs.append(seq)
        ys.append(y)

    if not seqs:
        print(f"[skip] {dir_path}: no windows produced (F={F}, T={WINDOW_SIZE}, stride={STRIDE})")
        return None, None

    X = np.stack(seqs, axis=0).astype(np.uint8)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y

