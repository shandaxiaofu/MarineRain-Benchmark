#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单进程 + 批处理：用 YOLOv8（COCO 预训练）检测文件夹内图片中的“船（boat）”。
- 在 NMS 阶段即限制类别为 boat（非事后过滤），更干净、速度也更快。
- CPU 默认运行；支持 HEIC/HEIF（需安装 pillow-heif）。
- 导出：总 CSV（逐框） + 逐图汇总 CSV；可选保存画框图。

依赖安装：
  pip install ultralytics pillow tqdm
  # 如需 HEIC/HEIF 支持（iPhone 原片）：
  pip install pillow-heif
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Any
import xml.etree.ElementTree as ET
import math

from PIL import Image, ImageDraw
from tqdm import tqdm


# ============ CONFIG（在此修改） ============
IMAGE_DIR = r"D:\ocean\detect\MRF"       # 输入图片目录（递归）
OUT_CSV   = r"./outputs/boats.csv"  # 检测结果 CSV（逐框）
DRAW_DIR  = r"D:/ocean/out/MRF"        # 画框输出目录；设为 "" 或 None 可关闭

# VOC 标注（XML）目录；如设置为有效路径，将在检测完成后基于 VOC 计算 boat 的 AP/mAP@0.5
VOC_ANN_DIR = r"D:/Monadepth/out_vis/VOC/Annotations"  # 例：r"D:/ocean/VOC/Annotations"

# 评估相关
EVAL_CLASS_NAME = "boat"            # VOC 标注里对应的类别名（区分大小写与否见下）
EVAL_CLASS_NAME_CASE_INSENSITIVE = True
IOU_THRESH_AP  = 0.1              # VOC2007 mAP@0.5
VOC2007_11POINT = True               # True: 11 点插值（VOC2007）；False: 积分法（VOC2010+）

DEVICE    = "cpu"                   # 你的机器无 CUDA：请设 'cpu'
MODEL     = "yolov8s.pt"            # 也可用 yolov8n.pt（更快更弱）
IMG_SIZE  = 1280                    # 640/960/1280；大分辨率更利于小目标
MIN_CONF  = 0.25                   # 低阈值提高召回
IOU       = 0.5                     # NMS IOU；保持默认或略高
MAX_DET   = 300                     # 每张图最大检测数
BATCH_SIZE = 2                      # CPU 环境建议 1~2；显存/内存吃紧就更小

WRITE_PER_IMAGE_SUMMARY = True      # 写出逐图统计 CSV
ENABLE_HEIC = True                  # 启用 HEIC/HEIF 支持（需要安装 pillow-heif）
DEBUG_LIST_FIRST = 5                # 启动时打印前 N 张图片路径；设 0 不打印
# ==========================================


# 支持的扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# 可选启用 HEIC/HEIF 支持
if ENABLE_HEIC:
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
        IMG_EXTS |= {".heic", ".heif"}
        print("[DEBUG] HEIC/HEIF enabled")
    except Exception as e:
        print("[DEBUG] HEIC not enabled:", e)


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def chunks(lst: List[str], batch: int) -> Iterable[List[str]]:
    if batch <= 0:
        batch = 1
    for i in range(0, len(lst), batch):
        yield lst[i:i + batch]


def draw_boxes(img_path: Path, boxes: List[Tuple[int, int, int, int, float]], out_dir: Path):
    """
    在图上画船框后保存。
    boxes: [(x1,y1,x2,y2,score), ...]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path).convert("RGB") as im:
        draw = ImageDraw.Draw(im, "RGBA")
        for (x1, y1, x2, y2, score) in boxes:
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0, 255), width=3)
            label = f"boat:{score:.2f}"
            tw = draw.textlength(label); th = 14
            draw.rectangle((x1, max(0, y1 - th - 6), x1 + tw + 8, y1), fill=(0, 255, 0, 100))
            draw.text((x1 + 4, max(0, y1 - th - 2)), label, fill=(0, 0, 0, 255))
        im.save(out_dir / img_path.name)


def get_cls_name(model, res, cls_id: int) -> str:
    """从模型/结果对象里取类别名，小写。兼容不同版本字段形态。"""
    name = ""
    if hasattr(model, "names"):
        # names 可能是 dict 或 list
        if isinstance(model.names, dict):
            name = str(model.names.get(int(cls_id), "")).lower()
        elif isinstance(model.names, (list, tuple)):
            try:
                name = str(model.names[int(cls_id)]).lower()
            except Exception:
                name = ""
    if not name and hasattr(res, "names"):
        if isinstance(res.names, dict):
            name = str(res.names.get(int(cls_id), "")).lower()
        elif isinstance(res.names, (list, tuple)):
            try:
                name = str(res.names[int(cls_id)]).lower()
            except Exception:
                name = ""
    return name


def _norm_key_from_path(p: str) -> str:
    """把路径规约为用于匹配的 key（文件名不含后缀）。"""
    try:
        return Path(p).stem
    except Exception:
        return str(p)


def _maybe_lower(s: str) -> str:
    return s.lower() if EVAL_CLASS_NAME_CASE_INSENSITIVE else s


def parse_voc_xml_for_class(xml_path: Path, target_class: str) -> List[Dict[str, Any]]:
    """
    解析单个 VOC XML，返回目标类的标注框列表。
    输出元素：{"bbox": (xmin,ymin,xmax,ymax), "difficult": bool}
    """
    boxes: List[Dict[str, Any]] = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="")
            if _maybe_lower(name) != _maybe_lower(target_class):
                continue
            difficult_text = obj.findtext("difficult", default="0")
            difficult = difficult_text.strip() in {"1", "true", "True"}
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = int(float(bnd.findtext("xmin", default="0")))
                ymin = int(float(bnd.findtext("ymin", default="0")))
                xmax = int(float(bnd.findtext("xmax", default="0")))
                ymax = int(float(bnd.findtext("ymax", default="0")))
            except Exception:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append({"bbox": (xmin, ymin, xmax, ymax), "difficult": difficult})
    except Exception:
        pass
    return boxes


def collect_voc_ground_truth(ann_dir: Path, image_keys: List[str], cls_name: str) -> Dict[str, Dict[str, Any]]:
    """
    收集 GT：返回 {image_key: {"boxes": [(x1,y1,x2,y2), ...], "difficult": [bool,...], "detected": [False,...]}}
    只统计指定类。
    """
    gt: Dict[str, Dict[str, Any]] = {}
    # 建立可用的 xml 映射：按 stem 匹配
    xml_map: Dict[str, Path] = {}
    for xml in ann_dir.glob("*.xml"):
        xml_map[xml.stem] = xml

    for key in image_keys:
        xml_path = xml_map.get(key)
        if xml_path is None:
            continue
        items = parse_voc_xml_for_class(xml_path, cls_name)
        if not items:
            continue
        boxes = [it["bbox"] for it in items]
        difficult = [bool(it["difficult"]) for it in items]
        detected = [False] * len(items)
        gt[key] = {"boxes": boxes, "difficult": difficult, "detected": detected}
    return gt


def iou_xyxy(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    if not precisions or not recalls:
        return 0.0
    # 排序并做 precision envelope
    pairs = sorted(zip(recalls, precisions))
    recalls_sorted = [p[0] for p in pairs]
    precisions_sorted = [p[1] for p in pairs]
    # precision envelope（从后往前取最大值）
    for i in range(len(precisions_sorted) - 2, -1, -1):
        precisions_sorted[i] = max(precisions_sorted[i], precisions_sorted[i + 1])
    if VOC2007_11POINT:
        ap = 0.0
        for t in [i / 10 for i in range(0, 11)]:
            p = 0.0
            for r, pr in zip(recalls_sorted, precisions_sorted):
                if r >= t:
                    p = max(p, pr)
            ap += p / 11.0
        return ap
    else:
        # VOC2010+ 积分法（对召回做分段积分）
        ap = 0.0
        mrec = [0.0] + recalls_sorted + [1.0]
        mpre = [0.0] + precisions_sorted + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        for i in range(1, len(mrec)):
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap


def evaluate_map_from_csv(csv_path: Path, ann_dir: Path, cls_name: str, iou_thresh: float = 0.5) -> Dict[str, float]:
    """
    仅评估单类（boat）：返回 {"AP@0.5": ap, "mAP@0.5": ap}
    """
    # 读取预测
    detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []  # (img_key, score, bbox)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _maybe_lower(row.get("label", "")) != _maybe_lower(cls_name):
                continue
            try:
                img_key = _norm_key_from_path(row["image_path"])  # 用文件名（无后缀）匹配
                x1 = int(float(row["x_min"]))
                y1 = int(float(row["y_min"]))
                x2 = int(float(row["x_max"]))
                y2 = int(float(row["y_max"]))
                score = float(row["score"]) if "score" in row else 0.0
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((img_key, score, (x1, y1, x2, y2)))

    # 排序（分数降序）
    detections.sort(key=lambda x: x[1], reverse=True)

    # 收集 GT
    image_keys = list({d[0] for d in detections})
    gt = collect_voc_ground_truth(ann_dir, image_keys, cls_name)
    npos = 0
    for item in gt.values():
        # VOC 评估：不把 difficult 计入正样本数
        npos += sum(1 for d in item["difficult"] if not d)

    tp = [0] * len(detections)
    fp = [0] * len(detections)

    for i, (img_key, score, box) in enumerate(detections):
        g = gt.get(img_key)
        if g is None:
            fp[i] = 1
            continue
        boxes_gt = g["boxes"]
        difficult = g["difficult"]
        detected = g["detected"]

        iou_max = 0.0
        jmax = -1
        for j, gt_box in enumerate(boxes_gt):
            iou = iou_xyxy(box, gt_box)
            if iou > iou_max:
                iou_max = iou
                jmax = j

        if iou_max >= iou_thresh:
            if difficult[jmax]:
                # 与 difficult 匹配：按 VOC 习惯，不计入 TP/FP（这里做成忽略，记为 FP=0；也可设为 FP）
                fp[i] = 0
                tp[i] = 0
            else:
                if not detected[jmax]:
                    tp[i] = 1
                    detected[jmax] = True
                else:
                    fp[i] = 1  # 重复匹配
        else:
            fp[i] = 1

    # 累加并计算 PR
    cum_tp = []
    cum_fp = []
    s_tp = 0
    s_fp = 0
    for i in range(len(detections)):
        s_tp += tp[i]
        s_fp += fp[i]
        cum_tp.append(s_tp)
        cum_fp.append(s_fp)
    precisions: List[float] = []
    recalls: List[float] = []
    for ct, cf in zip(cum_tp, cum_fp):
        precision = ct / max(1, (ct + cf))
        recall = ct / max(1, npos)
        precisions.append(precision)
        recalls.append(recall)

    ap = compute_ap(precisions, recalls)
    return {"AP@0.5": ap, "mAP@0.5": ap}


def main():
    img_root = Path(IMAGE_DIR)
    out_csv = Path(OUT_CSV)
    draw_dir = Path(DRAW_DIR) if DRAW_DIR else None

    if not img_root.exists():
        raise FileNotFoundError(f"输入路径不存在：{img_root}")

    paths = list_images(img_root)
    images = [str(p) for p in paths]
    print(f"[DEBUG] IMAGE_DIR={img_root} -> found {len(images)} images")
    if DEBUG_LIST_FIRST > 0:
        print("[DEBUG] first", min(DEBUG_LIST_FIRST, len(images)), ":", images[:DEBUG_LIST_FIRST])

    if not images:
        print("未找到图片。支持的后缀：", ", ".join(sorted(IMG_EXTS)))
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if draw_dir:
        draw_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    from ultralytics import YOLO
    model = YOLO(MODEL)

    # 找到 'boat' 类的 id，用于 classes 过滤（NMS 前过滤）
    # 兼容 names 为 dict 或 list
    boat_ids = []
    if hasattr(model, "names"):
        if isinstance(model.names, dict):
            boat_ids = [i for i, n in model.names.items() if str(n).lower() == "boat"]
        elif isinstance(model.names, (list, tuple)):
            for i, n in enumerate(model.names):
                if str(n).lower() == "boat":
                    boat_ids.append(i)

    if not boat_ids:
        print("[WARN] 未在模型类别表里找到 'boat'，将退化为事后按名称过滤。")

    # 打开结果 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["image_path", "label", "x_min", "y_min", "x_max", "y_max", "score"])

        # 逐图汇总 CSV（可选）
        if WRITE_PER_IMAGE_SUMMARY:
            out_sum = out_csv.with_name(out_csv.stem + "_per_image_summary.csv")
            fsum = open(out_sum, "w", newline="", encoding="utf-8")
            sum_writer = csv.writer(fsum)
            sum_writer.writerow(["image_path", "total_boxes_all", "boat_boxes", "error"])
        else:
            fsum = None; sum_writer = None

        total_written = 0

        # 批处理预测
        for batch_paths in tqdm(list(chunks(images, BATCH_SIZE)), desc="Predict"):
            # 预测参数；若能拿到 boat_ids，则在 NMS 前即限制类别
            predict_kwargs = dict(
                source=batch_paths,
                imgsz=IMG_SIZE,
                conf=MIN_CONF,
                iou=IOU,
                max_det=MAX_DET,
                device=DEVICE,
                verbose=False
            )
            if boat_ids:
                predict_kwargs["classes"] = boat_ids  # 只推理 boat

            try:
                results = model.predict(**predict_kwargs)
            except Exception as e:
                # 整个 batch 失败：写入错误信息
                if sum_writer:
                    for ip in batch_paths:
                        sum_writer.writerow([ip, 0, 0, f"batch_error:{e}"])
                continue

            # 解析每张图
            for ip, res in zip(batch_paths, results):
                try:
                    total_all = int(res.boxes.shape[0]) if res.boxes is not None else 0

                    vis_boxes = []
                    boat_cnt = 0
                    if res.boxes is not None and total_all > 0:
                        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                        scores = res.boxes.conf.cpu().numpy()
                        clses = res.boxes.cls.cpu().numpy().astype(int)

                        for (xyxy, score, cls_id) in zip(boxes_xyxy, scores, clses):
                            cls_name = get_cls_name(model, res, int(cls_id))

                            # 如果 classes 生效，则理论上这里只会有 boat；
                            # 若 boat_ids 为空（没找到类别名），则事后按名称过滤：
                            if boat_ids and cls_name != "boat":
                                # 极少数权重/版本差异情况下的兜底
                                continue
                            if not boat_ids and cls_name != "boat":
                                continue

                            boat_cnt += 1
                            x1, y1, x2, y2 = [int(round(v)) for v in xyxy.tolist()]
                            writer.writerow([ip, cls_name, x1, y1, x2, y2, f"{float(score):.4f}"])
                            total_written += 1

                            if draw_dir:
                                vis_boxes.append((x1, y1, x2, y2, float(score)))

                    # 可视化
                    if draw_dir and vis_boxes:
                        try:
                            draw_boxes(Path(ip), vis_boxes, draw_dir)
                        except Exception:
                            pass

                    if sum_writer:
                        sum_writer.writerow([ip, total_all, boat_cnt, ""])

                except Exception as e:
                    if sum_writer:
                        sum_writer.writerow([ip, 0, 0, f"{e}"])

        if fsum:
            fsum.close()

    print("完成！")
    print(f"- 结果 CSV：{out_csv}")
    if WRITE_PER_IMAGE_SUMMARY:
        print(f"- 每图汇总：{out_csv.with_name(out_csv.stem + '_per_image_summary.csv')}")
    if draw_dir:
        print(f"- 可视化输出：{draw_dir.resolve()}")
    print(f"- 写入框总数：{total_written}")

    # 如提供 VOC 标注目录，则执行评估
    if VOC_ANN_DIR:
        ann_dir = Path(VOC_ANN_DIR)
        if ann_dir.exists() and ann_dir.is_dir():
            try:
                metrics = evaluate_map_from_csv(out_csv, ann_dir, EVAL_CLASS_NAME, IOU_THRESH_AP)
                ap = metrics.get("AP@0.5", 0.0)
                print(f"[EVAL] {EVAL_CLASS_NAME} AP@{IOU_THRESH_AP:.2f} (VOC{'2007' if VOC2007_11POINT else '2010+'}): {ap:.4f}")
                # 写入评估结果
                out_metrics = out_csv.with_name(out_csv.stem + "_metrics.txt")
                with open(out_metrics, "w", encoding="utf-8") as fm:
                    fm.write(f"class,{EVAL_CLASS_NAME}\n")
                    fm.write(f"AP@{IOU_THRESH_AP:.2f},{ap:.6f}\n")
                    fm.write(f"mAP@{IOU_THRESH_AP:.2f},{ap:.6f}\n")
                print(f"- 评估结果：{out_metrics}")
            except Exception as e:
                print(f"[EVAL] 评估失败：{e}")
        else:
            print(f"[EVAL] 未找到 VOC 标注目录或无效：{ann_dir}")


if __name__ == "__main__":
    main()

