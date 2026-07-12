import cv2
import numpy as np
import glob
import os

# ========== 1. 读取带红框的参考图片 ==========
ref_path = r"K:\ocean-raindata\diff\rain\66.png"   # 你的“画好红框”的图
ref = cv2.imread(ref_path)
h, w = ref.shape[:2]

# ========== 2. 从参考图中提取所有红框 ==========
# OpenCV 默认是 BGR 排列，这里用一个简单的“接近红色”的阈值
# 视你画框的颜色/粗细可能要微调
lower_red = np.array([0, 0, 150])   # B G R
upper_red = np.array([80, 80, 255])

mask = cv2.inRange(ref, lower_red, upper_red)   # 红色像素区域
# 可选：做一下膨胀，连通成连续边框
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

# 找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x, y, ww, hh = cv2.boundingRect(c)
    # 过滤掉可能的噪声小框，比如宽高太小的
    if ww > 10 and hh > 10:
        boxes.append((x, y, x+ww, y+hh))

# 按从左到右、从上到下排序，方便命名
boxes.sort(key=lambda b: (b[1], b[0]))

print("检测到的红框数量：", len(boxes))
for i, b in enumerate(boxes):
    print(f"box {i}: {b}")

# ========== 3. 批量对其它图片画同样的框并裁剪 ==========
# 要处理的图片列表（都要和参考图同尺寸）
img_paths = sorted(glob.glob(r"K:\ocean-raindata\diff\MRF\66.png"))  # 自己改一下匹配规则

out_dir_crops = r"K:\ocean-raindata\diff\MRF"
out_dir_boxed = r"K:\ocean-raindata\diff\MRF"
os.makedirs(out_dir_crops, exist_ok=True)
os.makedirs(out_dir_boxed, exist_ok=True)

for path in img_paths:
    img = cv2.imread(path)
    if img.shape[:2] != (h, w):
        raise ValueError(f"尺寸不一致: {path}")

    base = os.path.splitext(os.path.basename(path))[0]

    # 备份一张用来画框的
    boxed = img.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # 画框（绿色，你也可以改成红色）
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 裁剪并保存框内区域
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out_dir_crops, f"{base}_crop{i+1}.png"), crop)

    # 保存“画好框”的整图，方便检查
    cv2.imwrite(os.path.join(out_dir_boxed, f"{base}_boxed.png"), boxed)

print("完成！所有裁剪结果在 'crops' 文件夹里，带框的图在 'boxed' 里。")
