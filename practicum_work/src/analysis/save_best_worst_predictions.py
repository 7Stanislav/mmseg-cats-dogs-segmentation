import os
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from PIL import Image
from tqdm import tqdm
import shutil

CONFIG_PATH = "configs/hyp4_deeplabv3p_r50.py"
CHECKPOINT_PATH = "work_dirs/hyp4_deeplabv3p_r50/best_mDice_iter_1000.pth"
TEST_IMG_DIR = "clean_v1/img/test"
TEST_LABEL_DIR = "clean_v1/labels/test"
OUTPUT_DIR = "practicum_work/supplementary/viz/qual_test_top_bottom"
NUM_SAVE = 5


def dice_score(pred, target, num_classes=3):
    dices = []
    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)

        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection) / union)

    return np.mean(dices)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = Config.fromfile(CONFIG_PATH)
    model = init_model(cfg, CHECKPOINT_PATH, device='cuda')

    results = []

    for fname in tqdm(os.listdir(TEST_IMG_DIR)):
        img_path = os.path.join(TEST_IMG_DIR, fname)
        label_path = os.path.join(TEST_LABEL_DIR, fname.replace(".jpg", ".png"))

        gt = np.array(Image.open(label_path))
        pred = inference_model(model, img_path)
        pred_mask = pred.pred_sem_seg.data.cpu().numpy()[0]

        score = dice_score(pred_mask, gt)
        results.append((fname, score))

    results.sort(key=lambda x: x[1])

    worst = results[:NUM_SAVE]
    best = results[-NUM_SAVE:]

    for name, _ in worst:
        shutil.copy(
            os.path.join(TEST_IMG_DIR, name),
            os.path.join(OUTPUT_DIR, f"worst_{name}")
        )

    for name, _ in best:
        shutil.copy(
            os.path.join(TEST_IMG_DIR, name),
            os.path.join(OUTPUT_DIR, f"best_{name}")
        )

    print("Saved best and worst examples.")


if __name__ == "__main__":
    main()
