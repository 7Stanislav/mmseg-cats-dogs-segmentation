import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def _find_image_for_stem(img_dir: Path, stem: str):
    # allow common image extensions
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read_label_u8(label_path: Path):
    # Read as single-channel without any palette conversion
    # PIL keeps indices for paletted PNG; converting to np gives indices.
    with Image.open(label_path) as im:
        im = im.convert("L") if im.mode not in ("L", "P") else im
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    # force uint8 (most datasets use uint8 indices)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _read_rgb(img_path: Path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return np.array(im)




def _save_overlay(rgb: np.ndarray, label: np.ndarray, out_path: Path, alpha: float = 0.45):
    # simple deterministic palette (works for any uint8 labels)
    lbl = label.astype(np.uint8)
    col = np.stack([
        (lbl * 37) % 255,
        (lbl * 17) % 255,
        (lbl * 97) % 255,
    ], axis=-1).astype(np.uint8)

    overlay = (rgb.astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="Path to train_dataset_for_students")
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--max-items", type=int, default=200)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    img_dir = data_root / "img" / args.split
    lbl_dir = data_root / "labels" / args.split

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Missing: {lbl_dir}")

    label_paths = sorted([p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".bmp", ".tif", ".tiff"]])
    if not label_paths:
        raise FileNotFoundError(f"No label files found in: {lbl_dir}")

    uniques_global = set()
    pixel_counts = {}  # value -> count (int)
    bad_samples = []
    total_pairs = 0
    overlays_saved = 0

    # For reporting mismatch pairs
    missing_img = 0
    unreadable_img = 0
    size_mismatch = 0
    single_value_masks = 0

    out_overlay_dir = Path("practicum_work") / "supplementary" / "viz" / "eda_samples" / args.split
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    for lbl_path in tqdm(label_paths, desc=f"Scan labels ({args.split})"):
        stem = lbl_path.stem
        img_path = _find_image_for_stem(img_dir, stem)
        if img_path is None:
            bad_samples.append({"stem": stem, "label": str(lbl_path), "image": None, "reason": "missing_image_for_label"})
            missing_img += 1
            continue

        try:
            label = _read_label_u8(lbl_path)
        except Exception as e:
            bad_samples.append({"stem": stem, "label": str(lbl_path), "image": str(img_path), "reason": f"label_read_error: {e}"})
            continue

        try:
            rgb = _read_rgb(img_path)
        except Exception as e:
            bad_samples.append({"stem": stem, "label": str(lbl_path), "image": str(img_path), "reason": f"image_read_error: {e}"})
            unreadable_img += 1
            continue

        total_pairs += 1

        if (rgb.shape[0] != label.shape[0]) or (rgb.shape[1] != label.shape[1]):
            bad_samples.append({
                "stem": stem,
                "label": str(lbl_path),
                "image": str(img_path),
                "reason": f"size_mismatch img={rgb.shape[1]}x{rgb.shape[0]} label={label.shape[1]}x{label.shape[0]}",
            })
            size_mismatch += 1
            continue

        u, c = np.unique(label, return_counts=True)
        for val, cnt in zip(u.tolist(), c.tolist()):
            uniques_global.add(int(val))
            pixel_counts[int(val)] = int(pixel_counts.get(int(val), 0) + int(cnt))

        if len(u) == 1:
            single_value_masks += 1
            bad_samples.append({
                "stem": stem,
                "label": str(lbl_path),
                "image": str(img_path),
                "reason": f"single_value_mask value={int(u[0])}",
            })

        if overlays_saved < args.max_items:
            out_path = out_overlay_dir / f"{stem}__overlay.png"
            try:
                _save_overlay(rgb, label, out_path)
                overlays_saved += 1
            except Exception as e:
                bad_samples.append({
                    "stem": stem,
                    "label": str(lbl_path),
                    "image": str(img_path),
                    "reason": f"overlay_save_error: {e}",
                })

    uniques_sorted = sorted(list(uniques_global))
    total_pixels = int(sum(pixel_counts.values()))
    # compute top by share
    top_items = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_items_pretty = [
        {"value": int(v), "count": int(cnt), "share": (float(cnt) / float(total_pixels) if total_pixels > 0 else 0.0)}
        for v, cnt in top_items
    ]

    report = {
        "split": args.split,
        "data_root": str(data_root),
        "img_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
        "total_label_files": int(len(label_paths)),
        "total_pairs_checked": int(total_pairs),
        "overlays_saved": int(overlays_saved),
        "uniques": uniques_sorted,
        "total_pixels_counted": total_pixels,
        "pixel_counts": {str(k): int(v) for k, v in sorted(pixel_counts.items(), key=lambda x: x[0])},
        "top_values": top_items_pretty,
        "bad_samples_count": int(len(bad_samples)),
        "bad_samples": bad_samples[:5000],  # safety cap
        "counters": {
            "missing_img": int(missing_img),
            "unreadable_img": int(unreadable_img),
            "size_mismatch": int(size_mismatch),
            "single_value_masks": int(single_value_masks),
        },
    }

    out_json = Path("practicum_work") / "supplementary" / "viz" / f"eda_report_{args.split}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== EDA SCAN REPORT ===")
    print(f"split: {args.split}")
    print(f"labels: {lbl_dir}")
    print(f"images: {img_dir}")
    print(f"label files: {len(label_paths)}")
    print(f"pairs checked: {total_pairs}")
    print(f"bad samples: {len(bad_samples)}")
    print(f"overlays saved: {overlays_saved} -> {out_overlay_dir}")
    print(f"unique label values: {uniques_sorted}")
    if 255 in uniques_global:
        print("NOTE: value 255 found (often used as ignore_index).")
    print("top values by pixel share:")
    for it in top_items_pretty:
        print(f"  value={it['value']}: share={it['share']:.4f} count={it['count']}")
    print(f"json report: {out_json}")


if __name__ == "__main__":
    main()
