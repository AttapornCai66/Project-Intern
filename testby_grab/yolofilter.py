import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm

# --- ตั้งค่า ---
INPUT_VIDEO_FOLDER = r"C:\Users\oppo0\OneDrive\Documents\GitHub\Project-Intern\Data\VDO" # <--- แก้ทางไปโฟลเดอร์ที่มีวิดีโอ
OUTPUT_FRAME_FOLDER = r"C:\Users\oppo0\OneDrive\Documents\GitHub\Project-Intern\Data\img"  # <--- ชื่อโฟลเดอร์ผลลัพธ์
CONFIDENCE_THRESHOLD = 0.6  # <--- ค่าความมั่นใจ (0-1) ตั้งไว้สูงหน่อยเพื่อเอาแต่ขวดชัดๆ
SAVE_EVERY_N_FRAME = 3  # <--- (เสริม) จากเฟรมที่มีขวด ให้เซฟทุกๆ กกี่เฟรม (ลดภาพซ้ำ)

# 1. โหลด Pre-trained YOLOv8 รุ่นจิ๋ว (รู้จักขวดอยู่แล้ว)
model = YOLO('yolov8n.pt')

# สร้างโฟลเดอร์สำหรับเก็บภาพผลลัพธ์
os.makedirs(OUTPUT_FRAME_FOLDER, exist_ok=True)

# 2. วนลูปอ่านทุกไฟล์วิดีโอในโฟลเดอร์
supported_formats = (".mp4", ".mov", ".avi")
video_files = [f for f in os.listdir(INPUT_VIDEO_FOLDER) if f.lower().endswith(supported_formats)]

print(f"พบวิดีโอ {len(video_files)} ไฟล์. กำลังเริ่มกรองขวด...")

for video_file in video_files:
    video_path = os.path.join(INPUT_VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    # Progress bar สำหรับแต่ละวิดีโอ
    with tqdm(total=total_frames, desc=f"กำลังประมวลผล {video_file}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # (เสริม) ลดภาพซ้ำซ้อน: ข้ามเฟรมถ้าไม่ได้ตาม N
            if frame_count % SAVE_EVERY_N_FRAME != 0:
                pbar.update(1)
                continue

            # 3. ให้ AI ดูว่าเฟรมนี้มีขวดไหม
            # classes=[39] คือรหัสคลาสสำหรับ 'bottle' ใน COCO Dataset
            results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD, classes=[39])

            for result in results:
                # ถ้าเจออย่างน้อย 1 ขวดในเฟรมนี้
                if len(result.boxes) > 0:
                    # 4. เซฟภาพเฟรมนั้นทันที
                    filename = f"{os.path.splitext(video_file)[0]}_fr{frame_count}.jpg"
                    filepath = os.path.join(OUTPUT_FRAME_FOLDER, filename)
                    cv2.imwrite(filepath, frame)
                    saved_count += 1

            pbar.update(1)

    cap.release()
    print(f"--> เสร็จสิ้น {video_file}: บันทึก {saved_count} เฟรมที่มีขวด.")

print(
    f"\n✅ กรองเสร็จสิ้น! ภาพขวดทั้งหมดอยู่ที่โฟลเดอร์ '{OUTPUT_FRAME_FOLDER}' พร้อมเอาไป Label ต่อบน Roboflow แล้วครับ")