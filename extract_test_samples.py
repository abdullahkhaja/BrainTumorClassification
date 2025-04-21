# extract_test_samples.py

import os
import shutil
import random
import torch
from torch.utils.data import random_split
from model.dataset import BrainTumorDataset, transform

def main():
    # 1) Repro for consistent splits
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # 2) Where kagglehub dumped your data
    data_root = os.path.expanduser(
        "~/.cache/kagglehub/datasets/rm1000/brain-tumor-mri-scans/versions/1"
    )

    # 3) Load the full dataset
    full_ds = BrainTumorDataset(root_dir=data_root, transform=transform)
    total = len(full_ds)

    # 4) Compute split sizes (must match train.py)
    train_sz = int(0.7 * total)
    val_sz   = int(0.15 * total)
    test_sz  = total - train_sz - val_sz

    # 5) Get the subsets
    train_ds, val_ds, test_ds = random_split(full_ds, [train_sz, val_sz, test_sz])

    # 6) Prepare output dir
    out_dir = os.path.join(os.getcwd(), "test_samples")
    os.makedirs(out_dir, exist_ok=True)

    # 7) Per‑class counters
    counters = {cls: 0 for cls in full_ds.class_names}

    # 8) Copy & rename
    for orig_idx in test_ds.indices:
        src_path = full_ds.images[orig_idx]
        label    = full_ds.labels[orig_idx]
        cls_name = full_ds.class_names[label]

        # bump counter
        counters[cls_name] += 1
        count = counters[cls_name]

        # build new filename
        ext = os.path.splitext(src_path)[1]  # e.g. ".jpg"
        dst_name = f"{cls_name}-{count}{ext}"
        dst_path = os.path.join(out_dir, dst_name)

        shutil.copy(src_path, dst_path)

    # 9) Summary
    total_copied = sum(counters.values())
    print(f"✅ Copied {total_copied} test images into `{out_dir}`:")
    for cls, cnt in counters.items():
        print(f"   • {cls}: {cnt}")

if __name__ == "__main__":
    main()
