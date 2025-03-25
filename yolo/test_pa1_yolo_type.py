import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO

# Danh sách các lớp theo thứ tự (chỉ số 0 đến 6)
class_names = [
    'TYPE 1 - P05', 
    'TYPE 2 - IQA1425', 
    'TYPE 2 - IQA1900', 
    'TYPE 2 - IQA3800', 
    'TYPE 2 - IQA950', 
    'TYPE 7 - L1', 
    'TYPE2-IQA2750A'
]

def enhance_image(image):
    """Enhance một ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.2
    beta = 10
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_eq = clahe.apply(adjusted)
    gaussian_blur = cv2.GaussianBlur(adaptive_eq, (3, 3), 0)
    blurred = cv2.GaussianBlur(gaussian_blur, (5, 5), 0)
    unsharp_mask = cv2.addWeighted(gaussian_blur, 1.2, blurred, -0.2, 0)
    sobelx = cv2.Sobel(unsharp_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(unsharp_mask, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs_x = cv2.convertScaleAbs(sobelx)
    sobel_abs_y = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(sobel_combined, kernel, iterations=1)
    return dilation

def process_and_overlay(original_image, enhanced_image, white_threshold=100, lower_white_threshold=120):
    """Tạo overlay từ ảnh đã enhance và đè lên ảnh gốc."""
    if enhanced_image is None:
        print("Error: Enhanced image is None")
        return None
        
    _, mask = cv2.threshold(enhanced_image, white_threshold, 255, cv2.THRESH_BINARY)
    _, lower_white_mask = cv2.threshold(enhanced_image, lower_white_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_white_mask))
    rgba_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGRA)
    rgba_image[:, :, 3] = mask
    rgba_image[mask > 0, 0:3] = [0, 0, 0]
    kernel = np.ones((3, 3), np.uint8)
    dilated_alpha = cv2.dilate(rgba_image[:, :, 3], kernel, iterations=1)
    rgba_image[:, :, 3] = dilated_alpha

    if original_image.shape[:2] != rgba_image.shape[:2]:
          rgba_image = cv2.resize(rgba_image, (original_image.shape[1], original_image.shape[0]))

    alpha_mask = rgba_image[:, :, 3] / 255.0
    alpha_mask_3channel = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)
    foreground = (rgba_image[:, :, :3] * alpha_mask_3channel).astype(np.uint8)
    background = (original_image * (1 - alpha_mask_3channel)).astype(np.uint8)
    final_result = cv2.add(foreground, background)
    return final_result

def encode_image_to_base64(image):
    """Chuyển đổi ảnh (NumPy array) thành chuỗi base64."""
    retval, buffer = cv2.imencode('.jpg', image)
    if retval:
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    else:
        print("Error: Could not encode image to base64.")
        return None

def count_pipes_for_class(image, model, expected_class, conf_threshold=0.34):
    """
    Chạy model YOLO trên ảnh (sau khi enhance và overlay),
    lọc các box có độ tin cậy >= conf_threshold và chỉ xét các box có lớp == expected_class.
    Trả về số lượng box tìm được và ảnh overlay đã vẽ bounding box.
    """
    if image is None:
        print("Error: Input image is None.")
        return 0, None

    enhanced_image = enhance_image(image)
    overlayed_image = process_and_overlay(image, enhanced_image)
    if overlayed_image is None:
        return 0, None

    results_model = model(overlayed_image)
    detected_boxes = results_model[0].boxes
    count = 0
    for box in detected_boxes:
        conf = box.conf[0].item()
        cls_pred = int(box.cls[0].item())
        if conf < conf_threshold or cls_pred != expected_class:
            continue
        count += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(overlayed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(overlayed_image, f'{count}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    base64_image = encode_image_to_base64(overlayed_image)
    return count, base64_image

def evaluate_yolo_per_class(model_path, base_folder, valid_extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Duyệt qua các thư mục con trong 'base_folder' (mỗi thư mục tương ứng với 1 loại) và đánh giá model YOLO.
    Với mỗi ảnh, ta đếm số đối tượng ground truth cho lớp đó (dựa trên file label)
    và số đối tượng được phát hiện bởi model (chỉ xét box có lớp tương ứng).
    Tính tỷ lệ đúng cho mỗi loại.
    """
    model = YOLO(model_path)
    per_class_stats = { cls: {"total": 0, "correct": 0} for cls in class_names }

    # Duyệt qua các thư mục con (mỗi thư mục tương ứng với một lớp)
    for idx, cls_name in enumerate(class_names):
        folder_path = os.path.join(base_folder, cls_name)
        if not os.path.exists(folder_path):
            print(f"Không tìm thấy thư mục: {folder_path}")
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_path = os.path.join(root, file)
                    label_path = os.path.splitext(image_path)[0] + ".txt"
                    if not os.path.exists(label_path):
                        continue

                    # Đọc file label và chỉ đếm các dòng có chỉ số lớp bằng expected (idx)
                    try:
                        with open(label_path, "r") as f:
                            lines = f.readlines()
                        gt_count = 0
                        for line in lines:
                            tokens = line.strip().split()
                            if not tokens:
                                continue
                            try:
                                label_cls = int(tokens[0])
                                if label_cls == idx:
                                    gt_count += 1
                            except:
                                continue
                    except Exception as e:
                        print(f"Lỗi khi đọc {label_path}: {e}")
                        continue

                    # Nếu không có ground truth nào cho lớp này, bỏ qua ảnh
                    if gt_count == 0:
                        continue

                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Lỗi khi đọc ảnh: {image_path}")
                        continue

                    detected_count, _ = count_pipes_for_class(image, model, expected_class=idx)
                    per_class_stats[cls_name]["total"] += 1
                    if detected_count == gt_count:
                        per_class_stats[cls_name]["correct"] += 1
                    print(f"Image: {file} | GT: {gt_count} | Detected: {detected_count} | "
                          f"Class: {cls_name} | Correct: {detected_count == gt_count}")

    # Tính toán và in ra tỷ lệ đúng cho mỗi loại
    for cls_name, stats in per_class_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"{cls_name}: Đã test {total} ảnh. Tỉ lệ đoán đúng: {accuracy:.2f}%")
        else:
            print(f"{cls_name}: Không có ảnh để test.")

    return per_class_stats

if __name__ == "__main__":
    # Đường dẫn tới model đã train (ví dụ: training_results_1m.pt)
    model_path = 'yolopan1.pt'
    # Đường dẫn tới thư mục "type" đã tạo ra
    base_folder = "type"
    stats = evaluate_yolo_per_class(model_path, base_folder)
