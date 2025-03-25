import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from attention_unet import build_attention_unet
import torch
from sklearn.cluster import DBSCAN  # Thêm thư viện clustering

# ------------------------------
# Cấu hình và load mô hình YOLOv11s và Attention UNet
# ------------------------------
model_yolo = YOLO("yolov11m2-train.pt")  # File mô hình detection đã được huấn luyện

class_names = {
    0: "TYPE 1 - P05",
    1: "TYPE 2 - IQA1425",
    2: "TYPE 2 - IQA1900",
    3: "TYPE 2 - IQA3800",
    4: "TYPE 2 - IQA950",
    5: "TYPE 7 - L1",
    6: "TYPE2-IQA2750A",
    7: "background"
}

num_classes = 8
input_size = (256, 256, 3)
model_unet = build_attention_unet(input_size, num_classes)

if os.path.exists("attention_unet_coco.h5"):
    model_unet.load_weights("attention_unet_coco.h5")
    print("Loaded Attention UNet weights.")
else:
    print("Warning: Attention UNet weights not found.")

# ------------------------------
# Hàm apply_nms: Loại bỏ các bbox trùng lặp (giữ nguyên)
# ------------------------------
def apply_nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    bboxes = np.array([b["bbox"] for b in boxes])
    scores = np.array([b["confidence"] for b in boxes])
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep_indices = torch.ops.torchvision.nms(bboxes_tensor, scores_tensor, iou_threshold)
    keep_indices = keep_indices.numpy()
    return [boxes[i] for i in keep_indices]

# ------------------------------
# Hàm process_yolo_results (giữ nguyên)
# ------------------------------
def process_yolo_results(results, conf_threshold=0.3):
    boxes = []
    for r in results:
        for box in r.boxes:
            if box.conf > conf_threshold:
                x_center, y_center, width, height = box.xywh[0]
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "confidence": confidence
                })
    return boxes

# ------------------------------
# Hàm adjust_for_crowded (giữ nguyên)
# ------------------------------
def adjust_for_crowded(predicted_boxes, crowded_threshold=50, eps=20):
    if len(predicted_boxes) <= crowded_threshold:
        return predicted_boxes
    centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in predicted_boxes])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = clustering.labels_
    unique_labels = set(labels)
    new_boxes = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        boxes_cluster = np.array([predicted_boxes[i] for i in indices])
        x1 = int(np.min(boxes_cluster[:, 0]))
        y1 = int(np.min(boxes_cluster[:, 1]))
        x2 = int(np.max(boxes_cluster[:, 2]))
        y2 = int(np.max(boxes_cluster[:, 3]))
        new_boxes.append([x1, y1, x2, y2])
    return new_boxes

# ------------------------------
# Hàm load_ground_truth_count (giữ nguyên)
# ------------------------------
def load_ground_truth_count(label_path, expected_class):
    gt_count = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            if tokens:
                try:
                    cls_val = int(tokens[0])
                    if cls_val == expected_class:
                        gt_count += 1
                except:
                    continue
    return gt_count

# ------------------------------
# Hàm pipeline_inference_with_unet_roi:
# 1. Dùng UNet để xác định ROI chứa vật thể (bằng cách chạy UNet trên toàn ảnh).
# 2. Dùng ROI đó để chạy YOLO và đếm đối tượng.
# 3. Nếu YOLO không đếm ra (yolo_count = 0), sử dụng kết quả của UNet.
# ------------------------------
def pipeline_inference_with_unet_roi(image_path, margin=5):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Không thể đọc ảnh {image_path}")
        return None

    # Bước 1: Chạy UNet trên toàn ảnh
    # Resize ảnh xuống 256x256 cho UNet
    img_resized = cv2.resize(orig_img, (256,256))
    img_input = img_resized.astype(np.float32)/255.0
    img_input = np.expand_dims(img_input, axis=0)
    pred_mask = model_unet.predict(img_input)[0, ..., 0]
    mask_bin = (pred_mask > 0.4).astype(np.uint8)*255
    # Upsample mask về kích thước gốc
    mask_upsampled = cv2.resize(mask_bin, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Tìm contours trên mask_upsampled để xác định ROI
    contours, _ = cv2.findContours(mask_upsampled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("UNet không phát hiện vùng chứa vật thể.")
        roi_bbox = None
    else:
        xs = []
        ys = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            xs.extend([x, x+w])
            ys.extend([y, y+h])
        roi_bbox = [max(min(xs)-margin,0), max(min(ys)-margin,0),
                    min(max(xs)+margin, orig_img.shape[1]-1), min(max(ys)+margin, orig_img.shape[0]-1)]
    
    # Tính unet_count: số đối tượng theo UNet (sử dụng connected components trên toàn mask_upsampled)
    num_labels, _ = cv2.connectedComponents(mask_upsampled)
    unet_count = num_labels - 1  # trừ background

    # Bước 2: Nếu có ROI, chạy YOLO trên ROI
    if roi_bbox is not None:
        x1,y1,x2,y2 = roi_bbox
        roi = orig_img[y1:y2, x1:x2]
        results_roi = model_yolo.predict(roi)
        boxes_roi = apply_nms(process_yolo_results(results_roi, conf_threshold=0.3), iou_threshold=0.3)
        yolo_count = len(boxes_roi)  # mỗi box YOLO đếm là 1 đối tượng
    else:
        yolo_count = 0

    # Bước 3: Lựa chọn kết quả
    # Nếu YOLO đếm ra > 0, lấy kết quả của YOLO; nếu không, dùng UNet
    final_count = yolo_count if yolo_count > 0 else unet_count

    print(f"Image: {os.path.basename(image_path)} | YOLO_count (ROI): {yolo_count} | UNet_count: {unet_count} | Final_count: {final_count}")
    return {
        "yolo_count": yolo_count,
        "unet_count": unet_count,
        "final_count": final_count,
        "roi_bbox": roi_bbox
    }

# ------------------------------
# Hàm evaluate_yolo_on_type_folder: Đánh giá model theo từng loại trong thư mục "type"
# ------------------------------
def evaluate_yolo_on_type_folder(base_folder="type", valid_extensions=(".jpg", ".jpeg", ".png", ".bmp"), iou_threshold=0.3):
    # Mapping từ tên thư mục sang chỉ số lớp (bỏ background)
    name_to_classid = {v: k for k, v in class_names.items() if k != 7}
    stats = {cls_name: {"total": 0, "correct": 0} for cls_name in name_to_classid.keys()}

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        # So sánh tên không phân biệt chữ hoa thường và bỏ khoảng trắng
        found = None
        for key in name_to_classid.keys():
            if key.lower().replace(" ", "") == folder.lower().replace(" ", ""):
                found = key
                break
        if found is None:
            print(f"Bỏ qua thư mục {folder} vì không khớp với danh sách class_names.")
            continue
        expected_class = name_to_classid[found]

        print(f"\n[INFO] Đang đánh giá cho loại: {folder} (class_id={expected_class})")
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
        for file in image_files:
            image_path = os.path.join(folder_path, file)
            label_path = os.path.splitext(image_path)[0] + ".txt"
            gt_count = load_ground_truth_count(label_path, expected_class)
            if gt_count == 0:
                print(f"[INFO] Ảnh {file} không có ground truth cho loại {folder}, bỏ qua.")
                continue

            result = pipeline_inference_with_unet_roi(image_path, margin=5)
            if result is None:
                continue

            final_count = result.get("final_count", 0)
            stats[folder]["total"] += 1
            if final_count == gt_count:
                stats[folder]["correct"] += 1
            print(f"Image: {file} | GT: {gt_count} | Final: {final_count} | Correct: {final_count==gt_count}")

    for cls_name, data in stats.items():
        total = data["total"]
        correct = data["correct"]
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"\n{cls_name}: Đã test {total} ảnh. Tỉ lệ đoán đúng: {accuracy:.2f}%")
        else:
            print(f"\n{cls_name}: Không có ảnh để test.")
    return stats

if __name__ == "__main__":
    evaluate_yolo_on_type_folder(base_folder="type", valid_extensions=(".jpg", ".jpeg", ".png", ".bmp"), iou_threshold=0.3)
