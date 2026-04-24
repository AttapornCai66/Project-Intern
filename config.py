# ============================================================
#  ⚙️  config.py — แก้ที่นี่ที่เดียว ใช้ได้ทุก script
# ============================================================

# 🔑 Roboflow vAJbag3YAQhy17MmbTXh
ROBOFLOW_API_KEY   = "vAJbag3YAQhy17MmbTXh"        #"IX2OWHwyaVaCqja3syDl"     # roboflow.com → Settings → API
ROBOFLOW_WORKSPACE = "project3-qfwc3"   # ดูจาก URL เช่น roboflow.com/my-workspace/...
ROBOFLOW_PROJECT   = "label_bottle_new"
ROBOFLOW_VERSION   = 2
ROBOFLOW_VERSIONDOWNLOAD = 'yolov8'  # ฟอร์แมตที่ต้องการดาวน์โหลด (เช่น "yolov8", "coco128", "voc", ฯลฯ)
# 🤖 YOLOv8 Model
# ตัวเลือก: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
# n = เร็วสุด/เล็กสุด  →  x = แม่นสุด/ใหญ่สุด
MODEL_SIZE = "yolov8s.pt"

# 🏋️ Training
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 2      # ลดเหลือ 8 ถ้า RAM/VRAM ไม่พอ
PATIENCE    = 20      # early stopping

# 📁 Output
_model_tag   = MODEL_SIZE.replace(".pt", "")   # เช่น yolov8s
PROJECT_NAME = "bottle_inspection"  # โฟลเดอร์หลักที่เก็บผลทุกโมเดล
RUN_NAME     = f"train_{_model_tag}"  

# 🖥️ Device: "0" = GPU แรก, "cpu" = CPU only, "0,1" = multi-GPU
DEVICE = "0"

# 🔍 Inference
CONF_THRESHOLD = 0.5
