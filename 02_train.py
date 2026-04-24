"""
02_train.py — Train YOLOv8 ด้วย dataset จาก Roboflow
"""
import os
from ultralytics import YOLO
from config import *


def get_dataset_path():
    if os.path.exists(".dataset_path"):
        with open(".dataset_path") as f:
            return f.read().strip()
    raise FileNotFoundError("❌ ไม่พบ dataset path — รัน 01_download.py ก่อน")


def train():
    dataset_path = get_dataset_path()
    data_yaml    = os.path.join(dataset_path, "data.yaml")

    # print("=" * 50)
    # print("🏋️  Training YOLOv8")
    # print("=" * 50)
    # print(f"Model   : {MODEL_SIZE}")
    # print(f"Data    : {data_yaml}")
    # print(f"Epochs  : {EPOCHS}")
    # print(f"Batch   : {BATCH_SIZE}")
    # print(f"ImgSize : {IMG_SIZE}")
    # print(f"Device  : {DEVICE}")
    # print()

    # โหลด pretrained weights
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        save=True,
        plots=True,       # สร้าง confusion_matrix, PR_curve, results.png
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
