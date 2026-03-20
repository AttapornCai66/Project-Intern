"""
========================================================
  STEP 1: แยก Frame จากวิดีโอ เพื่อสร้าง Dataset
  - รองรับ .mp4, .avi, .mov, .mkv
  - กรอง frame ที่คล้ายกันออก (ไม่เอา frame ซ้ำ)
  - แบ่ง train/val/test อัตโนมัติ
========================================================
"""

import cv2
import os
import random
import shutil
from pathlib import Path
import numpy as np

# ============================================================
# CONFIG — แก้ค่าตรงนี้
# ============================================================
VIDEO_PATHS = [
    "dataset/Videos/20251025_080202.mp4",   # ใส่ path วิดีโอของคุณ
    "videos/line2.avi",
]

OUTPUT_DIR      = "dataset/images"   # โฟลเดอร์เก็บ frame
FRAME_INTERVAL  = 10      # แยก 1 frame ทุกๆ N frame (10 = ~3fps จากวิดีโอ 30fps)
MIN_DIFF_SCORE  = 15.0    # ค่าต่ำสุด mean diff ระหว่าง frame (กรองซ้ำ)
TARGET_SIZE     = (640, 640)  # ขนาด frame ที่ต้องการ
SPLIT_RATIO     = (0.70, 0.20, 0.10)  # train / val / test
SEED            = 42


# ============================================================
# STEP 1: แยก Frame
# ============================================================
def extract_frames_from_video(video_path: str, output_dir: str) -> list:
    """
    อ่านวิดีโอและแยก frame ออกมา
    - ข้าม frame ที่คล้ายกันเกินไป (blur, static)
    - resize เป็น TARGET_SIZE
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ เปิดวิดีโอไม่ได้: {video_path}")
        return []

    video_name  = Path(video_path).stem
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n📹 วิดีโอ: {video_name}")
    print(f"   FPS: {fps:.1f} | Frames: {total_frames} | ประมาณ {total_frames/fps:.1f} วินาที")

    saved_frames = []
    prev_gray    = None
    frame_idx    = 0
    saved_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


        # แยกทุกๆ FRAME_INTERVAL
        if frame_idx % FRAME_INTERVAL == 0:

            # ตรวจสอบว่า frame ไม่ซ้ำกับ frame ก่อน
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                if diff < MIN_DIFF_SCORE:
                    frame_idx += 1
                    continue  # ข้าม frame ที่เหมือนกันมาก

            # Resize
            # frame_resized = cv2.resize(frame, TARGET_SIZE)
            frame_resized = frame
            # บันทึก
            filename = f"{video_name}_f{frame_idx:06d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            saved_frames.append(save_path)
            prev_gray = gray
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"   ✅ บันทึก {saved_count} frames (จาก {total_frames} frames ทั้งหมด)")
    return saved_frames


# ============================================================
# STEP 2: แบ่ง Train / Val / Test
# ============================================================
def split_frames(all_frames: list, base_output_dir: str):
    """แบ่ง frame เป็น train/val/test แบบ random"""

    random.seed(SEED)
    random.shuffle(all_frames)

    n       = len(all_frames)
    n_train = int(n * SPLIT_RATIO[0])
    n_val   = int(n * SPLIT_RATIO[1])

    splits = {
        "train": all_frames[:n_train],
        "val":   all_frames[n_train:n_train + n_val],
        "test":  all_frames[n_train + n_val:],
    }

    for split, frames in splits.items():
        split_dir = Path(base_output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for src in frames:
            dst = split_dir / Path(src).name
            shutil.move(src, dst)

    print(f"\n📊 แบ่ง Dataset:")
    print(f"   Train : {len(splits['train'])} frames")
    print(f"   Val   : {len(splits['val'])} frames")
    print(f"   Test  : {len(splits['test'])} frames")
    print(f"   รวม   : {n} frames")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("  🎬 Video Frame Extractor for YOLOv8 Training")
    print("=" * 55)

    # สร้างโฟลเดอร์ temp เก็บ frame ก่อนแบ่ง
    temp_dir = Path(OUTPUT_DIR) / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # แยก frame จากทุกวิดีโอ
    all_frames = []
    for video_path in VIDEO_PATHS:
        if not Path(video_path).exists():
            print(f"⚠️  ไม่พบไฟล์: {video_path}")
            continue
        frames = extract_frames_from_video(video_path, str(temp_dir))
        all_frames.extend(frames)

    if not all_frames:
        print("\n❌ ไม่พบ frame เลย กรุณาตรวจสอบ path วิดีโอ")
        return

    print(f"\n📦 รวมทั้งหมด: {len(all_frames)} frames จาก {len(VIDEO_PATHS)} วิดีโอ")

    # แบ่ง train/val/test
    split_frames(all_frames, OUTPUT_DIR)

    # ลบโฟลเดอร์ temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 55)
    print("  ✅ เสร็จสิ้น! ขั้นตอนถัดไป:")
    print("  1. เปิด Roboflow หรือ LabelImg")
    print(f"  2. Label ภาพใน: {OUTPUT_DIR}/train/")
    print("  3. Export เป็น YOLO format")
    print("  4. รัน 2_train_yolov8.py")
    print("=" * 55)


if __name__ == "__main__":
    main()