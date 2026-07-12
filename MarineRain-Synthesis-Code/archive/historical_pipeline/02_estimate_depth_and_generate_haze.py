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
img_path = Path(r'D:\ocean\input')
depth_path = Path(r'D:\ocean\mask_m_s')
hazy_path = Path(r'D:\ocean\haze_m_o')
# -----------------雾气强度控制因子----------------------#
fog_strength = 1.1
# ----------------雾气颜色 (浅灰色雾气)-------------------#
fog_color = np.array([200, 200, 200], dtype=np.uint8)
# ------------------------------------------------------#
#   可选择的model: 'mono_640x192'、'stereo_640x192'、'mono+stereo_640x192'、
#   'mono_1024x320'、'stereo_1024x320'、'mono+stereo_1024x320'、
# ------------------------------------------------------#
model = "models/mono+stereo_1024x320"
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

        depth_map = (depth_map_normalized * 255).astype(np.uint8)
        # 保存深度图 - 统一使用png格式
        new_filename = depth_path / Path(img).stem  # 移除原扩展名
        new_filename = new_filename.with_suffix('.png')  # 统一保存为png格式
        cv2.imwrite(str(new_filename), depth_map)

        pbar.update(1)

# -----------------------生成雾图----------------------------------#
with tqdm(total=len(imglist), desc=('雾图生成')) as pbar:
    for filename in os.listdir(img_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(img_path, filename)
            
            # 修改这里：深度图文件名与输入图像主文件名相同，但扩展名为.png
            depth_filename = Path(filename).stem + '.png'  # 保留主文件名，改为.png扩展名
            depthmap = os.path.join(depth_path, depth_filename)

            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"无法读取原始图像: {image_path}")
                continue
                
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_map = cv2.imread(depthmap, cv2.IMREAD_GRAYSCALE)
            
            # 反转深度图，使得白色区域（近处）雾气浓，黑色区域（远处）雾气稀薄
            if depth_map is None:
                print(f"跳过文件: {filename}，因为无法读取对应的深度图: {depthmap}")
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
