from ultralytics import YOLO
import numpy as np

# 加载模型（你训练好的）
model = YOLO(r"runs\detect\train7\weights\best.pt")

def garbage_image_recognize(standard_image: np.ndarray):
    """
    标准化接口：garbage_image_recognize（完全按文档）
    输入：预处理后的标准图像 ndarray
    输出：{name, confidence, bbox}
    """
    # 推理
    results = model(standard_image, conf=0.1)

    # 无目标
    if len(results[0].boxes) == 0:
        return {
            "name": None,
            "confidence": 0.0,
            "bbox": None,
            "msg": "未识别到垃圾目标"
        }

    # 取置信度最高的一个
    box = results[0].boxes[0]
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names[cls_id]
    bbox = box.xyxy[0].cpu().numpy().tolist()

    return {
        "name": name,
        "confidence": round(conf, 2),
        "bbox": bbox,
        "msg": "识别成功"
    }

if __name__ == "__main__":
    from preprocess import image_capture_preprocess
    img = image_capture_preprocess("test.jpg")
    res = garbage_image_recognize(img)
    print("接口返回：", res)