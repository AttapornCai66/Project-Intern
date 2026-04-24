"""
02_train.py — Train RT-DETR ด้วย dataset จาก Roboflow
"""
import os
from ultralytics import RTDETR
from config import *


def get_dataset_path():
    if os.path.exists(".dataset_path"):
        with open(".dataset_path") as f:
            return f.read().strip()
    raise FileNotFoundError("❌ ไม่พบ dataset path — รัน 01_download.py ก่อน")


def train():
    dataset_path = get_dataset_path()
    data_yaml    = os.path.join(dataset_path, "data.yaml")

    # โหลด pretrained weights (RT-DETR-l หรือ RT-DETR-x)
    model = RTDETR(MODEL_SIZE)  # เช่น "rtdetr-l.pt" หรือ "rtdetr-x.pt"

    # Train
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        optimizer='AdamW',
        lr0=0.0001,        # RT-DETR ใช้ LR ต่ำกว่า YOLO
        lrf=0.01,          # Final LR = lr0 * lrf
        cos_lr=True,       # Cosine LR scheduler
        # warmup_epochs=3,   # Warmup ช่วยให้ transformer เสถียรขึ้น
        # weight_decay=0.0001,
        # save=True,
        plots=True,
        verbose=True,
    )

    best_weights = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    print("\n✅ Training complete!")
    print(f"   Best weights : {best_weights}")
    print(f"   Results dir  : {PROJECT_NAME}/{RUN_NAME}/")

    # บันทึก weights path
    with open(".best_weights", "w") as f:
        f.write(best_weights)

    return results


if __name__ == "__main__":
    train()
    print("\n✅ Done! ต่อไปรัน: python 03_evaluate.py")