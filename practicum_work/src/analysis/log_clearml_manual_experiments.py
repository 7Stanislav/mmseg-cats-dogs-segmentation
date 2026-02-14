from clearml import Task

PROJECT = "mmseg-cats-dogs-segmentation"
TAGS = ["manual", "practicum"]

experiments = [
    {
        "name": "hyp1_segformer_b0_manual",
        "metrics": [("mDice", "val", 0.5736, 1000)],
        "notes": "Manual log from README: SegFormer-B0, CE+Dice, best mDice (val) at iter=1000."
    },
    {
        "name": "hyp2_segformer_b0_wce_manual",
        "metrics": [("mDice", "val", 0.5908, 2000)],
        "notes": "Manual log from README: SegFormer-B0, weighted CE, best mDice (val) at iter=2000."
    },
    {
        "name": "hyp3_segformer_b0_wce_soft_manual",
        "metrics": [("mDice", "val", 0.5991, 3000)],
        "notes": "Manual log from README: SegFormer-B0, soft weighted CE, best mDice (val) at iter=3000."
    },
    {
        "name": "hyp4_deeplabv3p_r50_manual",
        "metrics": [
            ("mDice", "test", 0.8978, 1),
            ("Dice", "background", 0.9878, 1),
            ("Dice", "cat", 0.8753, 1),
            ("Dice", "dog", 0.8302, 1),
        ],
        "notes": "Manual log from README: DeepLabV3+ R50, test metrics."
    },
]

def main():
    print("Creating ClearML manual experiments...")
    created = []
    for exp in experiments:
        task = Task.init(project_name=PROJECT, task_name=exp["name"], tags=TAGS)
        task.set_comment(exp["notes"])
        logger = task.get_logger()
        for title, series, value, it in exp["metrics"]:
            logger.report_scalar(title=title, series=series, value=float(value), iteration=int(it))
        # ensure upload of scalars
        task.flush(wait_for_uploads=True)
        url = task.get_output_log_web_page()
        created.append((exp["name"], url))
        task.close()

    print("\nDONE. Links:")
    for name, url in created:
        print(f"{name}: {url}")

if __name__ == "__main__":
    main()
