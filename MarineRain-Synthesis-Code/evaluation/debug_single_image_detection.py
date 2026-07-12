from ultralytics import YOLO
import cv2, os

IMG = r"D:/ocean/input/141.jpg"   # ← 换成你的一张图
MODEL = "yolov8s.pt"              # s 更稳，n 也行
IMGSZ = 1280
CONF  = 0.05

print("Env check:")
import torch, ultralytics
print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("  ultralytics:", ultralytics.__version__)
print("  exists:", os.path.exists(IMG))
im = cv2.imread(IMG)
print("  cv2.imread:", None if im is None else im.shape)

m = YOLO(MODEL)
res = m.predict(source=IMG, imgsz=IMGSZ, conf=CONF, device="cpu", verbose=True)
r = res[0]

n_boxes = 0 if r.boxes is None else r.boxes.shape[0]
print("boxes:", n_boxes)
if n_boxes:
    names = r.names if hasattr(r, "names") else m.names
    for xyxy, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int)):
        x1,y1,x2,y2 = [int(v) for v in xyxy.tolist()]
        label = names.get(int(cls), str(cls)).lower()
        print(f"{label} {conf:.3f} [{x1},{y1},{x2},{y2}]")

# 可视化看看
m.predict(source=IMG, imgsz=IMGSZ, conf=CONF, device="cpu", save=True, project="runs", name="debug_one", exist_ok=True)
print("Saved vis to runs/debug_one")
