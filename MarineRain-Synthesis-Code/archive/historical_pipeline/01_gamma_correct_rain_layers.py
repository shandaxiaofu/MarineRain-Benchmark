
# #!/usr/bin/env python3
# """
# batch_gamma_contrast.py
# -----------------------
# 批量为雨线图片做自适应 Gamma 压暗 + 对比度保护 (CLAHE)。

# 用法（命令行）：
#     python batch_gamma_contrast.py --in_dir /path/to/rain_imgs --out_dir /path/to/output
# 可选参数：
#     --exts      支持的图片扩展名，逗号分隔 (默认: jpg,png,jpeg)
#     --clip      CLAHE clipLimit (默认: 2.0)
#     --grid      CLAHE tileGridSize 单元格大小 (默认: 8)
#     --min_gamma 最小 Gamma (默认: 1.2)
#     --max_gamma 最大 Gamma (默认: 2.0)
#     --target_l  目标平均亮度 L* (0-255, 默认: 110)

# 脚本逻辑：
# 1. 计算每张图的平均亮度 (灰度均值)。
# 2. 根据亮度插值得到自适应 Gamma：
#        gamma = clip((target_l * 2) / max(mean, 1), min_gamma, max_gamma)
# 3. 先在 LAB 颜色空间对 L 通道做 CLAHE，保护对比度。
# 4. 然后对整个图像做 Gamma 压暗。
# 5. 保存到输出目录，保持原文件名。
# """

# import cv2, os, argparse, glob, numpy as np
# from pathlib import Path

# def clahe_l_channel(img_bgr, clip=2.0, grid=8):
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
#     l_eq = clahe.apply(l)
#     merged = cv2.merge((l_eq, a, b))
#     return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# def adjust_gamma(image, gamma):
#     inv = 1.0 / gamma
#     table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
#     return cv2.LUT(image, table)

# def process_image(path, out_path, clip, grid, min_gamma, max_gamma, target_l):
#     img = cv2.imread(str(path))
#     if img is None:
#         print(f"[WARN] 读取失败: {path}")
#         return
#     # Step 1: CLAHE
#     img_clahe = clahe_l_channel(img, clip=clip, grid=grid)
#     # Step 2: adaptive gamma
#     mean = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY).mean()
#     gamma = (target_l * 2) / max(mean, 1)  # 简单反比关系
#     gamma = np.clip(gamma, min_gamma, max_gamma)
#     img_gamma = adjust_gamma(img_clahe, gamma)
#     cv2.imwrite(str(out_path), img_gamma)
#     print(f"[OK] {path.name}  mean={mean:.1f}  gamma={gamma:.2f}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--in_dir', required=True, help='输入图片文件夹')
#     parser.add_argument('--out_dir', required=True, help='输出文件夹')
#     parser.add_argument('--exts', default='jpg,png,jpeg', help='支持的扩展名')
#     parser.add_argument('--clip', type=float, default=2.0, help='CLAHE clipLimit')
#     parser.add_argument('--grid', type=int, default=8, help='CLAHE tileGridSize')
#     parser.add_argument('--min_gamma', type=float, default=1.2)
#     parser.add_argument('--max_gamma', type=float, default=2.0)
#     parser.add_argument('--target_l', type=float, default=110, help='目标平均亮度')
#     args = parser.parse_args()

#     in_dir = Path(args.in_dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     exts = ['.' + e.lower() for e in args.exts.replace(',', ' ').split()]
#     files = [p for ext in exts for p in in_dir.rglob(f'*{ext}')]
#     print(f'共 {len(files)} 张图片，开始处理...')
#     for p in files:
#         out_path = out_dir / p.name
#         process_image(p, out_path, args.clip, args.grid,
#                       args.min_gamma, args.max_gamma, args.target_l)

# if __name__ == '__main__':
#     main()
#python gamma.py --in_dir D:\Monadepth\test_gamma --out_dir D:\Monadepth\test_gamma\output --clip 2.5 --grid 8 --min_gamma 2.5 --max_gamma 2.5 --target_l 105
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os, glob, pathlib

# ============= ① 亮斑抑制函数（可放脚本最前） =================
HOT_V_MIN   = 220   # ≥该值认为“过亮”
RGB_DIFF_TH = 10    # 最大值-最小值 ≥该值认为“彩点”

def suppress_hotspots(bgr):
    """
    将过亮且带彩度的小块“降亮+去色”，避免刺眼亮斑。
    bgr : uint8 BGR 图像 (numpy.ndarray)
    return : 处理后的同尺寸 BGR 图像
    """
    # 1) 判定掩膜：既太亮又有彩度
    mask = ((bgr.max(axis=2) >= HOT_V_MIN) & (bgr.ptp(axis=2) >= RGB_DIFF_TH)).astype(np.uint8)
    if mask.any():
        # 2) 膨胀 3×3，免得只改一个像素留下硬边
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        # 3) 把亮斑改成柔和灰（取自身均值×0.6）
        gray = (bgr[mask>0].mean(axis=1, keepdims=True) * 0.6).astype(np.uint8)
        bgr[mask>0] = np.repeat(gray, 3, axis=1)
    return bgr
# ============================================================


# ======================= ② 参数区 ============================
gamma    = 2.5          # >1 越暗
v_scale  = 0.76         # HSV 缩放因子（用 method='hsv' 时生效）
method   = 'gamma'      # 'gamma' 或 'hsv'

in_dir   = r'I:\rainsteak'        # 输入文件夹
out_dir  = r'D:\Monadepth\test_gamma\output2'    # 输出文件夹
# ============================================================

# 自动建输出目录
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

# 预先生成 gamma LUT（只用一次，CPU 零成本）
table = (np.linspace(0, 1, 256) ** gamma * 255).clip(0, 255).astype(np.uint8)

# 支持的图片扩展名
exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')

# ======================= ③ 主循环 ===========================
for pattern in exts:
    for img_path in glob.glob(os.path.join(in_dir, pattern)):
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] 读不到 {img_path}')
            continue

        # ------ 调暗：保持你原来的两种分支 ------
        if method == 'gamma':
            darker = cv2.LUT(img, table)
            suffix = f'_g{gamma}.png'
        elif method == 'hsv':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 2] = (hsv[..., 2] * v_scale).clip(0, 255)
            darker = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            suffix = f'_v{v_scale}.png'
        else:
            raise ValueError('method 只能是 "gamma" 或 "hsv"')

        # ------ ★★ 新增：亮斑抑制 ★★ ------
        darker = suppress_hotspots(darker)

        # ------ 写盘 ------
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, base + suffix)
        cv2.imwrite(out_path, darker)
        print(f'[OK] {img_path}  →  {out_path}')

print('全部处理完成！')