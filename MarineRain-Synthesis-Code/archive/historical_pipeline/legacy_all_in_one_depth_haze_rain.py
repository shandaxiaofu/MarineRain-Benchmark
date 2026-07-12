import os
import torch
import cv2
from torchvision import transforms
import numpy as np
from pathlib import Path
import matplotlib
 
matplotlib.use('TkAgg')
from tqdm import tqdm
from utils.resnet_encoder import ResnetEncoder
from utils.depth_decoder import DepthDecoder
 
 
# -----------------输入图像、深度图、雾图路径--------------#
img_path = Path(r'H:\LHP\gt\train')
depth_path = Path(r'D:\Monadepth\LHP\DEPTH')
hazy_path = Path(r'D:\Monadepth\LHP\haze')
# -----------------雾气强度控制因子----------------------#
fog_strength = 1.0
# ----------------雾气颜色 (浅灰色雾气)-------------------#
fog_color = np.array([200, 200, 200], dtype=np.uint8)
# ------------------------------------------------------#
#   可选择的model: 'mono_640x192'、'stereo_640x192'、'mono+stereo_640x192'、
#   'mono_1024x320'、'stereo_1024x320'、'mono+stereo_1024x320'、
# ------------------------------------------------------#
model = "models/mono_1024x320"
# ------------------------生成深度图------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------加载模型----------------------#
encoder_path = os.path.join(model, "encoder.pth")
depth_decoder_path = os.path.join(model, "depth.pth")
 
encoder = ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
 
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()
 
depth_decoder = DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()
 
imglist = os.listdir(img_path)
with tqdm(total=len(imglist), desc=('深度图转换')) as pbar:
    for img in imglist:
        full_path = img_path / img
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_width, original_height = image.shape[:2]
        image = cv2.resize(image, (feed_width, feed_height), interpolation=cv2.INTER_LANCZOS4)
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)
 
        with torch.no_grad():
            features = encoder(image)
            outputs = depth_decoder(features)
 
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_width, original_height), mode="bilinear", align_corners=False)
 
        depth_map = disp_resized.squeeze().cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        # # 可视化深度图
        # plt.imshow(depth_map_normalized, cmap='plasma')
        # plt.colorbar()
        # plt.title('Estimated Depth Map')
        # plt.show()
 
        depth_map = (depth_map_normalized * 255).astype(np.uint8)
        # 保存深度图
        new_filename = depth_path / img
        new_filename = new_filename.with_suffix('.png')  # 可以自己更改深度图的格式，默认为png
        cv2.imwrite(str(new_filename), depth_map)
 
        pbar.update(1)
 
# -----------------------生成雾图----------------------------------#
with tqdm(total=len(imglist), desc=('雾图生成')) as pbar:
    for filename in os.listdir(img_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(img_path, filename)
            depthmap = str(depth_path / Path(filename).with_suffix('.png'))
 
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_map = cv2.imread(depthmap, cv2.IMREAD_GRAYSCALE)
            # 反转深度图，使得白色区域（近处）雾气浓，黑色区域（远处）雾气稀薄
            if depth_map is None or original_image is None:
                print(f"跳过文件: {filename}，因为无法读取对应的深度图或原始图像")
                print('请检查输入图像与深度图是否属于同一类型，如均为.png或.jpg，深度图的格式默认为.png，可自行更改其类型，详情请查看代码注释部分')
                continue
 
            depth_map_normalized = depth_map.astype(np.float32) / 255
 
            depth_map_inverted = 1 - depth_map_normalized
            # 雾气强度基于反转后的深度图，应用强度因子控制
            fog_intensity_map = depth_map_inverted * fog_strength
            # 限制雾气浓度的最大值为 1，避免过度曝光
            fog_intensity_map = np.clip(fog_intensity_map, 0, 1)
            # 创建雾气层
            fog_layer = np.ones_like(original_image, dtype=np.float32) * fog_color
            # 将雾气强度与原始图像结合
            foggy_image = original_image * (1 - fog_intensity_map[:, :, np.newaxis]) + \
                          fog_layer * fog_intensity_map[:, :, np.newaxis]
            # 将结果转换为 uint8 类型
            foggy_image = np.clip(foggy_image, 0, 255).astype(np.uint8)
 
            output_path = os.path.join(hazy_path, filename)
            foggy_image_bgr = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)  # 转回 BGR 格式以保存
            cv2.imwrite(output_path, foggy_image_bgr)
 
        pbar.update(1)




# ----------------------- 叠加雨线图 ----------------------------------#
import random
import math

def get_rain_streak(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Rainy image not found: {filepath}")
    return cv2.imread(filepath).astype(np.float32) / 255.0

def crop_rain_streak(rain_img, target_height, target_width):
    h, w = rain_img.shape[:2]
    if h < target_height or w < target_width:
        scale = max(target_height/h, target_width/w) * 0.5
        rain_img = cv2.resize(rain_img, (int(w*scale), int(h*scale)))
        h, w = rain_img.shape[:2]
    y = random.randint(0, h - target_height)
    x = random.randint(0, w - target_width)
    return rain_img[y:y+target_height, x:x+target_width]

def add_rain_to_image(image, depth_map, rain_img):
    target_height, target_width = image.shape[:2]
    rain_cropped = crop_rain_streak(rain_img, target_height, target_width)
    if len(depth_map.shape) == 2:
        depth_map = np.stack([depth_map]*3, axis=-1)
    depth_map = cv2.resize(depth_map, (target_width, target_height))
    alpha = random.uniform(0.6, 0.8)
    beta = random.uniform(1.6, 1.9)
    k = np.exp(-beta * depth_map)
    result = image / 255.0 + alpha * rain_cropped * k
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# 雨线图路径配置
rainy_root = r"I:\rain_sy\rain4"
rain_files = [f for f in os.listdir(rainy_root) if f.lower().endswith(('.jpg', '.png'))]

# 雨图输出目录
rainy_output_path = Path(hazy_path) / "fog_rain"
os.makedirs(rainy_output_path, exist_ok=True)

with tqdm(total=len(imglist), desc=('雾雨图合成')) as pbar:
    for filename in imglist:
        try:
            rain_file = random.choice(rain_files)
            rain_img = get_rain_streak(os.path.join(rainy_root, rain_file))
            
            fog_img_path = hazy_path / filename
            foggy_image = cv2.imread(str(fog_img_path))
            foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)
            
            depth_img_path = depth_path / Path(filename).with_suffix('.png')
            depth_map = cv2.imread(str(depth_img_path), cv2.IMREAD_GRAYSCALE)
            if depth_map is None or foggy_image is None:
                print(f"跳过 {filename}，因雾图或深度图读取失败")
                continue
            depth_map = depth_map.astype(np.float32) / 255.0

            rainy_foggy_img = add_rain_to_image(foggy_image, depth_map, rain_img)
            rainy_foggy_img_bgr = cv2.cvtColor(rainy_foggy_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rainy_output_path / filename), rainy_foggy_img_bgr)
        except Exception as e:
            print(f"⚠️ 跳过 {filename}: {e}")
        pbar.update(1)
