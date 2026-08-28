"""
训练脚本：分别运行 FloodNet 和 CASlandslides 微调
CPU 训练优化：imgsz=320, batch=16, epochs=30
支持从中断处恢复训练
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.image_preprocess import (
    train_floodnet, train_landslide, train_rescuenet, train_floodimg_detect,
    evaluate_on_floodnet_val, evaluate_on_caslandslides,
    MODEL_DIR, ImageSegmenter, prepare_floodnet_yolo_dataset,
    prepare_caslandslides_yolo_dataset, prepare_rescuenet_yolo_dataset,
    prepare_floodimg_yolo_dataset, create_yolo_dataset_yaml,
    create_yolo_det_dataset_yaml
)

# 训练参数
EPOCHS = 30
IMGSZ = 320
BATCH = 16
DEVICE = "cpu"

# ========== 1. FloodNet 训练（支持恢复） ==========
print("=" * 60)
print("FloodNet 内涝分割模型训练")
print("=" * 60)

flood_last_pt = os.path.join(MODEL_DIR, "floodnet_train", "floodnet_exp", "weights", "last.pt")
flood_best_pt = os.path.join(MODEL_DIR, "floodnet_train", "floodnet_exp", "weights", "best.pt")
flood_target = os.path.join(MODEL_DIR, "flood.pt")

if os.path.exists(flood_last_pt):
    last_epoch = 0
    try:
        import torch
        ckpt = torch.load(flood_last_pt, map_location="cpu", weights_only=False)
        last_epoch = ckpt.get("epoch", 0) or 0
        print(f"检测到已有检查点，已完成 {last_epoch}/{EPOCHS} epochs")
    except Exception as e:
        print(f"读取检查点失败: {e}，将从头训练")

    if last_epoch >= EPOCHS:
        print(f"训练已完成，跳过 FloodNet 训练")
        if os.path.exists(flood_best_pt) and not os.path.exists(flood_target):
            import shutil
            shutil.copy2(flood_best_pt, flood_target)
            print(f"模型权重已保存到: {flood_target}")
    else:
        from ultralytics import YOLO
        yolo_dir = os.path.join(MODEL_DIR, "yolo_floodnet")
        yaml_path = create_yolo_dataset_yaml(yolo_dir, "flood")
        model = YOLO(flood_last_pt)
        remaining = EPOCHS - last_epoch
        print(f"恢复训练，剩余 {remaining} epochs")
        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=10,
            project=os.path.join(MODEL_DIR, "floodnet_train"),
            name="floodnet_exp",
            exist_ok=True,
            amp=True,
            resume=True,
        )
        if os.path.exists(flood_best_pt):
            import shutil
            shutil.copy2(flood_best_pt, flood_target)
            print(f"模型权重已保存到: {flood_target}")
else:
    t0 = time.time()
    try:
        train_floodnet(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE)
        print(f"FloodNet 训练完成，耗时: {(time.time()-t0)/60:.1f} 分钟")
    except Exception as e:
        print(f"FloodNet 训练失败: {e}")
        import traceback
        traceback.print_exc()

# ========== 2. CASlandslides 训练（支持恢复） ==========
print("\n" + "=" * 60)
print("CASlandslides 滑坡分割模型训练")
print("=" * 60)

casl_last_pt = os.path.join(MODEL_DIR, "casl_train", "casl_exp", "weights", "last.pt")
casl_best_pt = os.path.join(MODEL_DIR, "casl_train", "casl_exp", "weights", "best.pt")
casl_target = os.path.join(MODEL_DIR, "landslide.pt")

if os.path.exists(casl_last_pt):
    try:
        import torch
        ckpt = torch.load(casl_last_pt, map_location="cpu", weights_only=False)
        last_epoch = ckpt.get("epoch", 0) or 0
        print(f"检测到已有检查点，已完成 {last_epoch}/{EPOCHS} epochs")
    except:
        last_epoch = 0

    if last_epoch >= EPOCHS:
        print(f"训练已完成，跳过 CASlandslides 训练")
        if os.path.exists(casl_best_pt) and not os.path.exists(casl_target):
            import shutil
            shutil.copy2(casl_best_pt, casl_target)
            print(f"模型权重已保存到: {casl_target}")
    else:
        from ultralytics import YOLO
        yolo_dir = os.path.join(MODEL_DIR, "yolo_landslide")
        yaml_path = create_yolo_dataset_yaml(yolo_dir, "landslide")
        model = YOLO(casl_last_pt)
        remaining = EPOCHS - last_epoch
        print(f"恢复训练，剩余 {remaining} epochs")
        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=10,
            project=os.path.join(MODEL_DIR, "casl_train"),
            name="casl_exp",
            exist_ok=True,
            amp=True,
            resume=True,
        )
        if os.path.exists(casl_best_pt):
            import shutil
            shutil.copy2(casl_best_pt, casl_target)
            print(f"模型权重已保存到: {casl_target}")
else:
    t0 = time.time()
    try:
        train_landslide(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE)
        print(f"CASlandslides 训练完成，耗时: {(time.time()-t0)/60:.1f} 分钟")
    except Exception as e:
        print(f"CASlandslides 训练失败: {e}")
        import traceback
        traceback.print_exc()

# ========== 3. RescueNet 训练（支持恢复） ==========
print("\n" + "=" * 60)
print("RescueNet 洪水分割模型训练（CC BY-NC-ND）")
print("=" * 60)

rescuenet_last_pt = os.path.join(MODEL_DIR, "rescuenet_train", "rescuenet_exp", "weights", "last.pt")
rescuenet_best_pt = os.path.join(MODEL_DIR, "rescuenet_train", "rescuenet_exp", "weights", "best.pt")
rescuenet_target = os.path.join(MODEL_DIR, "flood_rescuenet.pt")

if os.path.exists(rescuenet_last_pt):
    try:
        import torch
        ckpt = torch.load(rescuenet_last_pt, map_location="cpu", weights_only=False)
        last_epoch = ckpt.get("epoch", 0) or 0
        print(f"检测到已有检查点，已完成 {last_epoch}/{EPOCHS} epochs")
    except:
        last_epoch = 0

    if last_epoch >= EPOCHS:
        print(f"训练已完成，跳过 RescueNet 训练")
        if os.path.exists(rescuenet_best_pt) and not os.path.exists(rescuenet_target):
            import shutil
            shutil.copy2(rescuenet_best_pt, rescuenet_target)
            print(f"模型权重已保存到: {rescuenet_target}")
    else:
        from ultralytics import YOLO
        yolo_dir = os.path.join(MODEL_DIR, "yolo_rescuenet")
        yaml_path = create_yolo_dataset_yaml(yolo_dir, "flood")
        model = YOLO(rescuenet_last_pt)
        remaining = EPOCHS - last_epoch
        print(f"恢复训练，剩余 {remaining} epochs")
        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=10,
            project=os.path.join(MODEL_DIR, "rescuenet_train"),
            name="rescuenet_exp",
            exist_ok=True,
            amp=True,
            resume=True,
        )
        if os.path.exists(rescuenet_best_pt):
            import shutil
            shutil.copy2(rescuenet_best_pt, rescuenet_target)
            print(f"模型权重已保存到: {rescuenet_target}")
else:
    t0 = time.time()
    try:
        train_rescuenet(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE)
        print(f"RescueNet 训练完成，耗时: {(time.time()-t0)/60:.1f} 分钟")
    except Exception as e:
        print(f"RescueNet 训练失败: {e}")
        import traceback
        traceback.print_exc()

# ========== 4. FloodIMG 训练（支持恢复） ==========
print("\n" + "=" * 60)
print("FloodIMG 洪水目标检测模型训练")
print("=" * 60)

floodimg_last_pt = os.path.join(MODEL_DIR, "floodimg_train", "floodimg_exp", "weights", "last.pt")
floodimg_best_pt = os.path.join(MODEL_DIR, "floodimg_train", "floodimg_exp", "weights", "best.pt")
floodimg_target = os.path.join(MODEL_DIR, "floodimg_detect.pt")

if os.path.exists(floodimg_last_pt):
    try:
        import torch
        ckpt = torch.load(floodimg_last_pt, map_location="cpu", weights_only=False)
        last_epoch = ckpt.get("epoch", 0) or 0
        print(f"检测到已有检查点，已完成 {last_epoch}/{EPOCHS} epochs")
    except:
        last_epoch = 0

    if last_epoch >= EPOCHS:
        print(f"训练已完成，跳过 FloodIMG 训练")
        if os.path.exists(floodimg_best_pt) and not os.path.exists(floodimg_target):
            import shutil
            shutil.copy2(floodimg_best_pt, floodimg_target)
            print(f"模型权重已保存到: {floodimg_target}")
    else:
        from ultralytics import YOLO
        yolo_dir = os.path.join(MODEL_DIR, "yolo_floodimg")
        yaml_path = create_yolo_det_dataset_yaml(yolo_dir, {})
        model = YOLO(floodimg_last_pt)
        remaining = EPOCHS - last_epoch
        print(f"恢复训练，剩余 {remaining} epochs")
        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=10,
            project=os.path.join(MODEL_DIR, "floodimg_train"),
            name="floodimg_exp",
            exist_ok=True,
            amp=True,
            resume=True,
        )
        if os.path.exists(floodimg_best_pt):
            import shutil
            shutil.copy2(floodimg_best_pt, floodimg_target)
            print(f"模型权重已保存到: {floodimg_target}")
else:
    t0 = time.time()
    try:
        train_floodimg_detect(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE)
        print(f"FloodIMG 训练完成，耗时: {(time.time()-t0)/60:.1f} 分钟")
    except Exception as e:
        print(f"FloodIMG 训练失败: {e}")
        import traceback
        traceback.print_exc()

# ========== 5. 评估 ==========
print("\n" + "=" * 60)
print("训练完成，开始评估")
print("=" * 60)

print("\n[FloodNet Val 评估]")
try:
    flood_report = evaluate_on_floodnet_val(num_samples=-1)
    print(json.dumps(flood_report, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"评估失败: {e}")

print("\n[CASlandslides 评估]")
try:
    casl_report = evaluate_on_caslandslides(num_samples=-1)
    print(json.dumps(casl_report, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"评估失败: {e}")

print("\n" + "=" * 60)
print("所有训练和评估完成")
print("=" * 60)