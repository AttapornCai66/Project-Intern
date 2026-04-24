"""
01_download.py — ตรวจสอบ environment + download dataset จาก Roboflow
"""
import subprocess, sys, os, yaml
import torch
from config import *


def check_env():
    print("=" * 50)
    print("🔧 Environment Check")
    print("=" * 50)

    # Python version
    print(f"Python  : {sys.version.split()[0]}")

    # PyTorch + CUDA
    print(f"PyTorch : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA    : {'✅ ' + torch.cuda.get_device_name(0) if cuda_ok else '❌ Not found (จะใช้ CPU)'}")

    if not cuda_ok:
        print("\n⚠️  ไม่พบ GPU — training จะช้ากว่า GPU มาก")
        print("   ถ้ามี NVIDIA GPU ให้ติดตั้ง CUDA: https://developer.nvidia.com/cuda-downloads")

    # nvidia-smi (optional)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GPU Info: {result.stdout.strip()}")
    except FileNotFoundError:
        pass

    print()


def download_dataset():
    print("=" * 50)
    print("📥 Downloading Dataset from Roboflow")
    print("=" * 50)

    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    # ดาวน์โหลดในฟอร์แมต YOLOv8
    dataset = version.download(ROBOFLOW_VERSIONDOWNLOAD)

    # บันทึก path ไว้ใช้ script ถัดไป
    with open(".dataset_path", "w") as f:
        f.write(dataset.location)

    # อ่าน class info
    with open(f"{dataset.location}/data.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"\n✅ Dataset saved to : {dataset.location}")
    print(f"   Classes ({cfg['nc']}) : {cfg['names']}")

    # นับจำนวนภาพ
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset.location, split, "images")
        if os.path.exists(img_dir):
            n = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
            print(f"   {split:5s} images : {n}")

    return dataset


if __name__ == "__main__":
    check_env()
    download_dataset()
    print("\n✅ Done! ต่อไปรัน: python 02_train.py")
