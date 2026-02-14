import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]
LBL_EXTS = [".png", ".bmp", ".tif", ".tiff"]


def find_img(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_label_u8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        # keep palette indices if present
        if im.mode not in ("L", "P"):
            im = im.convert("L")
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8, copy=False)


def load_imagenet_labels() -> list[str]:
    # torchvision weights provide category names
    weights = MobileNet_V3_Small_Weights.DEFAULT
    return list(weights.meta["categories"])


def predict_cat_dog(model, preprocess, labels, img_path: Path):
    # returns: ("cat"|"dog"|"unknown", confidence_float, top_label_str)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    x = preprocess(im).unsqueeze(0)  # 1x3xHxW
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    top_label = labels[int(idx)]
    conf = float(conf.item())

    s = top_label.lower()
    # crude but practical mapping for ImageNet labels
    cat_keys = ["cat", "kitten", "persian", "siamese", "tabby", "tiger cat", "egyptian cat"]
    dog_keys = ["dog", "puppy", "terrier", "retriever", "shepherd", "husky", "hound", "mastiff", "poodle", "spaniel"]

    is_cat = any(k in s for k in cat_keys)
    is_dog = any(k in s for k in dog_keys)

    if is_cat and not is_dog:
        return "cat", conf, top_label
    if is_dog and not is_cat:
        return "dog", conf, top_label
    return "unknown", conf, top_label


def swap_1_2(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    m1 = out == 1
    m2 = out == 2
    out[m1] = 2
    out[m2] = 1
    return out


def save_mask(mask_u8: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8, mode="L").save(out_path)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out-root", required=True, type=str, help="Where to write cleaned dataset")
    ap.add_argument("--min_fg_frac", type=float, default=0.002, help="Min non-bg fraction to consider mask non-empty")
    ap.add_argument("--conf_th", type=float, default=0.25, help="Min classifier confidence to use cat/dog decision")
    ap.add_argument("--dry-run", action="store_true", help="Do not write outputs, only report")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    img_dir = data_root / "img" / args.split
    lbl_dir = data_root / "labels" / args.split
    out_root = Path(args.out_root)
    out_img_dir = out_root / "img" / args.split
    out_lbl_dir = out_root / "labels" / args.split

    lbl_paths = sorted([p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() in LBL_EXTS])

    # model
    weights = MobileNet_V3_Small_Weights.DEFAULT
    labels = load_imagenet_labels()
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    stats = {
        "split": args.split,
        "total_labels": 0,
        "usable_pairs": 0,
        "missing_img": 0,
        "tiny_or_empty_mask": 0,
        "classifier_used": 0,
        "unknown_species": 0,
        "conf_th": args.conf_th,
        "min_fg_frac": args.min_fg_frac,
        "counts": {
            "species_cat_main1": 0,
            "species_cat_main2": 0,
            "species_dog_main1": 0,
            "species_dog_main2": 0,
        },
        "flagged_for_manual": [],   # list of stems
        "examples": [],             # small list for sanity
    }

    per_item = []  # for later decisions

    for lbl_path in tqdm(lbl_paths, desc=f"Scan+classify ({args.split})"):
        stats["total_labels"] += 1
        stem = lbl_path.stem
        img_path = find_img(img_dir, stem)
        if img_path is None:
            stats["missing_img"] += 1
            continue

        mask = read_label_u8(lbl_path)
        h, w = mask.shape[:2]
        total = h * w
        cnt1 = int((mask == 1).sum())
        cnt2 = int((mask == 2).sum())
        fg = cnt1 + cnt2
        fg_frac = fg / total

        # main label among {1,2}
        main = 1 if cnt1 >= cnt2 else 2

        if fg_frac < args.min_fg_frac:
            stats["tiny_or_empty_mask"] += 1
            stats["flagged_for_manual"].append({"stem": stem, "reason": f"tiny_fg fg_frac={fg_frac:.6f}"})
            per_item.append((stem, img_path, lbl_path, fg_frac, main, "unknown", 0.0, "tiny_fg"))
            continue

        species, conf, top_label = predict_cat_dog(model, preprocess, labels, img_path)

        use = (species in ("cat", "dog")) and (conf >= args.conf_th)
        if use:
            stats["classifier_used"] += 1
            key = f"species_{species}_main{main}"
            stats["counts"][key] += 1
        else:
            if species == "unknown":
                stats["unknown_species"] += 1
            stats["flagged_for_manual"].append({"stem": stem, "reason": f"low_conf_or_unknown species={species} conf={conf:.3f} top={top_label}"})

        # store
        per_item.append((stem, img_path, lbl_path, fg_frac, main, species, conf, top_label))

        if len(stats["examples"]) < 15:
            stats["examples"].append({
                "stem": stem,
                "fg_frac": fg_frac,
                "cnt1": cnt1, "cnt2": cnt2, "main": main,
                "species": species, "conf": conf, "top_label": top_label
            })

    # Decide global mapping using confident samples
    # We want: cat->? (1 or 2), dog->other
    c1 = stats["counts"]["species_cat_main1"]
    c2 = stats["counts"]["species_cat_main2"]
    d1 = stats["counts"]["species_dog_main1"]
    d2 = stats["counts"]["species_dog_main2"]

    mapping = {"cat": None, "dog": None}
    # simplest: choose cat label by majority
    if (c1 + c2) > 0:
        mapping["cat"] = 1 if c1 >= c2 else 2
        mapping["dog"] = 2 if mapping["cat"] == 1 else 1

    stats["usable_pairs"] = len(per_item)
    stats["mapping_inferred"] = mapping
    stats["note"] = "Mapping inferred from ImageNet classifier on images + dominant label in mask. Requires manual review for flagged samples."

    # Write cleaned dataset if mapping inferred and not dry-run
    actions = {"copied": 0, "swapped": 0, "skipped": 0}
    if not args.dry_run and mapping["cat"] is not None:
        for (stem, img_path, lbl_path, fg_frac, main, species, conf, top_label) in tqdm(per_item, desc=f"Write cleaned ({args.split})"):
            # copy image always (if we include sample)
            # strategy:
            # - if tiny_fg -> skip (remove from cleaned)
            # - if confident cat/dog and main != mapping[species] -> swap 1<->2 in mask
            # - else keep as is
            if isinstance(top_label, str) and top_label == "tiny_fg":
                actions["skipped"] += 1
                continue
            if fg_frac < args.min_fg_frac:
                actions["skipped"] += 1
                continue

            mask = read_label_u8(lbl_path)

            do_swap = False
            if species in ("cat", "dog") and conf >= args.conf_th:
                target = mapping[species]
                if target is not None and main != target:
                    do_swap = True

            copy_file(img_path, out_img_dir / img_path.name)
            if do_swap:
                mask2 = swap_1_2(mask)
                save_mask(mask2, out_lbl_dir / f"{stem}.png")
                actions["swapped"] += 1
            else:
                # normalize save to png
                save_mask(mask, out_lbl_dir / f"{stem}.png")
                actions["copied"] += 1

    stats["write_actions"] = actions

    out_report = Path("practicum_work") / "supplementary" / "viz" / f"label_fix_report_{args.split}.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== LABEL FIX REPORT ===")
    print(f"split: {args.split}")
    print(f"total labels: {stats['total_labels']}")
    print(f"usable pairs: {stats['usable_pairs']}")
    print(f"missing_img: {stats['missing_img']}")
    print(f"tiny_or_empty_mask: {stats['tiny_or_empty_mask']}")
    print(f"classifier_used (conf>={args.conf_th}): {stats['classifier_used']}")
    print(f"unknown/low_conf flagged: {len(stats['flagged_for_manual'])}")
    print(f"inferred mapping: {stats['mapping_inferred']}")
    print("counts (cat/dog vs main label):", stats["counts"])
    print(f"report: {out_report}")
    if not args.dry_run:
        print("write_actions:", actions)


if __name__ == "__main__":
    main()
