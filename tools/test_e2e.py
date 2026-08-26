"""
端到端验证脚本
测试 process_image 三个任务（flood / landslide / flood_detect）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.preprocess_api import process_image, map_to_bn_states


def test_flood():
    print("=" * 60)
    print("Test 1: process_image(task='flood') - 回归测试")
    print("=" * 60)
    img = r"H:\dev\disaster-data\image_datasets\floodnet\extracted\FloodNet-Supervised_v1.0\val\val-org-img\7819.jpg"
    if not os.path.exists(img):
        print(f"SKIP: 图片不存在 {img}")
        return
    result = process_image(img, task="flood")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    bn = map_to_bn_states(flood_area_m2=result.get("积水面积_m2"))
    print("→ BN映射:", json.dumps(bn, ensure_ascii=False, indent=2))
    needed = {"积水面积_m2", "淹没占比", "灾情等级", "推理耗时_s"}
    assert needed.issubset(result.keys()), f"缺少字段: {needed - result.keys()}"
    print("PASS: 字段完整\n")


def test_flood_detect():
    print("=" * 60)
    print("Test 2: process_image(task='flood_detect')")
    print("=" * 60)
    img = r"H:\dev\disaster-data\image_datasets\floodimg\Flood Images\flood_0.jpg"
    if not os.path.exists(img):
        print(f"SKIP: 图片不存在 {img}")
        return
    result = process_image(img, task="flood_detect")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    bn = map_to_bn_states(flood_detected_objects=result.get("洪水目标数"))
    print("→ BN映射:", json.dumps(bn, ensure_ascii=False, indent=2))
    needed = {"洪水目标数", "最大洪水框面积比例", "类别分布", "推理耗时_s"}
    assert needed.issubset(result.keys()), f"缺少字段: {needed - result.keys()}"
    print("PASS: 字段完整\n")


def test_landslide():
    print("=" * 60)
    print("Test 3: process_image(task='landslide') - 回归测试")
    print("=" * 60)
    img = r"H:\dev\disaster-data\models\yolo_landslide\images\val\LongxiheSAT1558.tif"
    if not os.path.exists(img):
        print(f"SKIP: 图片不存在 {img}")
        return
    result = process_image(img, task="landslide")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    bn = map_to_bn_states(landslide_area_m2=result.get("滑坡面积_m2"))
    print("→ BN映射:", json.dumps(bn, ensure_ascii=False, indent=2))
    needed = {"滑坡面积_m2", "灾情等级", "推理耗时_s"}
    assert needed.issubset(result.keys()), f"缺少字段: {needed - result.keys()}"
    print("PASS: 字段完整\n")


if __name__ == "__main__":
    test_flood()
    test_flood_detect()
    test_landslide()
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)