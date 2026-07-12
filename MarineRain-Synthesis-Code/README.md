# MarineRain synthesis code

本目录是 MarineRain 数据合成代码的 GitHub 整理版本。当前版本已经将论文中报告的公式和参数设为权威实现，并提供一个唯一的端到端命令行入口：

```text
generate_marinerain.py
```

历史脚本仍然原样保存在 `archive/historical_pipeline/` 中，用于追溯早期实验，但不再作为推荐运行入口。

## 1. 论文权威合成模型

主流程严格实现以下归一化图像形成模型：

```text
R_gc = R_orig ^ gamma
T    = exp(-beta * d)
O    = T * B + L * (1 - T) + T * R_gc
```

其中：

- `B`：干净背景图，数值范围 `[0, 1]`；
- `R_orig`：裁剪后的真实雨线层；
- `R_gc`：Gamma 矫正后的雨线层；
- `d`：由 Monodepth2 视差图反转并逐图归一化得到的相对深度；
- `T`：深度相关透射率；
- `L`：归一化大气光；
- `O`：最终合成的雨雾图像。

论文默认参数已经写入命令行默认值：

| 参数 | 默认值 |
|---|---:|
| Gamma `gamma` | `3.5` |
| 散射系数 `beta` | 从 `[1.0, 1.2]` 均匀采样 |
| 大气光 `L` | 从 `[0.8, 1.0]` 均匀采样 |
| 随机种子 | `1234` |

主流程不再使用历史代码中的线性雾化、固定 `200/255` 大气光、独立雨线强度 `alpha` 或 `[0.9, 1.1]` 的旧 `beta` 范围。

## 2. 目录结构

```text
MarineRain-Synthesis-Code/
├── generate_marinerain.py       # 唯一推荐入口：参数解析
├── marinerain_pipeline.py       # 深度提取、Gamma 和统一大气模型
├── environments.txt             # 参考 Python/pip 环境
├── CODE_AUDIT.md                # 历史代码与论文对齐记录
├── FILE_RENAME_MAP.md           # 原文件到归档文件的映射
├── .gitignore
│
├── utils/                       # Monodepth2 网络结构
│   ├── depth_decoder.py
│   ├── layers.py
│   └── resnet_encoder.py
│
├── models/                      # 本地放置模型权重，不提交 Git
│   └── README.md
│
├── archive/historical_pipeline/ # 未改动的历史实验脚本
├── evaluation/                  # YOLOv8 船舶检测评估工具
└── tools/                       # 论文图像区域裁剪工具
```

## 3. 创建运行环境

建议使用 Python 3.10：

```bash
cd "/path/to/MarineRain-Synthesis-Code"
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environments.txt
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

`environments.txt` 同时覆盖核心合成脚本、历史脚本和 `evaluation/` 中的依赖。当前版本只做了静态依赖整理，没有按照用户要求执行原始样例复现验证。

## 4. 放置 Monodepth2 权重

主流程至少需要：

```text
models/
└── mono+stereo_1024x320/
    ├── encoder.pth
    └── depth.pth
```

也可以通过 `--model-dir` 指向其他包含这两个文件的 Monodepth2 模型目录。

模型权重没有复制到本目录，也没有在本次处理中补充许可证或权重发布方案。

## 5. 准备输入目录

干净背景和雨线层分别放入两个目录，例如：

```text
data/
├── clean/
│   ├── sea_0001.jpg
│   └── sea_0002.jpg
└── rain_layers/
    ├── rain_0001.png
    └── rain_0002.png
```

支持的格式为：

```text
.jpg .jpeg .png .bmp .tif .tiff .webp
```

默认递归搜索子目录，可通过 `--no-recursive` 关闭。

## 6. 运行完整合成流程

从项目根目录执行：

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --model-dir models/mono+stereo_1024x320
```

该命令会依次完成：

1. 加载一次 Monodepth2 编码器和深度解码器；
2. 对每张干净图预测视差；
3. 将视差逐图归一化并反转为相对深度 `d`；
4. 随机选择雨线图并缩放、随机裁剪到背景尺寸；
5. 使用 `gamma=3.5` 计算 `R_gc`；
6. 采样 `beta` 和大气光 `L`；
7. 在同一个函数内计算 `T` 和完整大气图像形成公式；
8. 保存对齐的干净图、雨雾图、相对深度图和参数元数据。

## 7. 常用参数

生成每张背景的多个随机版本：

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --samples-per-image 2
```

明确使用 CPU：

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --device cpu
```

保存经过 Gamma 矫正和裁剪的雨线层：

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --save-gamma-rain
```

查看全部参数：

```bash
python generate_marinerain.py --help
```

关键可配置项包括：

```text
--gamma
--beta-min / --beta-max
--airlight-min / --airlight-max
--samples-per-image
--seed
--device auto|cpu|cuda|mps
--save-depth / --no-save-depth
--save-gamma-rain / --no-save-gamma-rain
--output-format png|jpg
```

如果需要严格复现论文报告设置，不要覆盖 `gamma`、`beta` 和 `airlight` 的默认值。

## 8. 输出结构

默认输出为：

```text
outputs/marinerain/
├── clean/
│   └── 000001_sea_0001_s01.png
├── rainy/
│   └── 000001_sea_0001_s01.png
├── relative_depth/
│   └── 000001_sea_0001_s01.png
└── metadata.csv
```

如果启用 `--save-gamma-rain`，还会生成：

```text
gamma_rain/
```

`metadata.csv` 为每一个样本记录：

- 干净图和雨线图的相对来源路径；
- 输出文件名；
- Gamma；
- 实际采样的 `beta`；
- 实际采样的大气光 `L`；
- 随机种子；
- 模型目录；
- 相对深度定义；
- 使用的图像形成公式。

相对深度图以 16-bit PNG 保存。它是 `1 - normalized disparity`，并不是米制或其他物理单位的绝对深度。

## 9. 随机性和重复运行

相同输入文件排序、相同模型、相同依赖环境和相同 `--seed` 会复用相同的：

- 雨线图选择；
- 随机裁剪位置；
- `beta`；
- 大气光 `L`。

默认要求输出目录中不存在 `metadata.csv`，以避免无意覆盖。确实需要重写同一目录时，可显式传入 `--overwrite`；该选项不会自动删除目录中的旧文件。

## 10. 本次没有执行的事项

按照用户要求，本次没有：

- 补充或确认第三方 Monodepth2 许可证；
- 发布、上传或建立模型权重下载方案；
- 使用原始输入和历史输出进行数值或视觉结果复现对比。

因此，代码已完成公式、参数、入口和配置层面的统一，并通过静态检查；模型和数据层面的端到端结果仍需在后续单独验证。
