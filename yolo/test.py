from ultralytics import YOLO
import cv2
import numpy as np
import base64

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


def process_and_overlay(original_image, enhanced_image, white_threshold=70, lower_white_threshold=50):
    """Tạo overlay từ ảnh đã enhance và đè lên ảnh gốc."""
    enhanced_img = enhanced_image  # Sử dụng trực tiếp ảnh đã enhance
    if enhanced_img is None:
        print(f"Error: Enhanced image is None")
        return None
        
    _, mask = cv2.threshold(enhanced_img, white_threshold, 255, cv2.THRESH_BINARY)
    _, lower_white_mask = cv2.threshold(enhanced_img, lower_white_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_white_mask))
    rgba_image = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGRA)
    rgba_image[:, :, 3] = mask
    rgba_image[mask > 0, 0:3] = [0, 0, 0]
    kernel = np.ones((3, 3), np.uint8)
    dilated_alpha = cv2.dilate(rgba_image[:, :, 3], kernel, iterations=1)
    rgba_image[:, :, 3] = dilated_alpha

    # Resize PNG nếu cần (nhưng không cần nếu kích thước đã giống nhau)
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
    retval, buffer = cv2.imencode('.jpg', image)  # Hoặc .png, tùy định dạng bạn muốn
    if retval:
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    else:
        print("Error: Could not encode image to base64.")
        return None
    
def count_pipes(image_base64, path_yolo):
    """Đếm số lượng ống và trả về số lượng, ảnh đã vẽ bounding box dưới dạng base64."""
    model = YOLO(path_yolo)
    
    # Giải mã chuỗi Base64 thành mảng byte
    image_data = base64.b64decode(image_base64)
    # Chuyển đổi dữ liệu byte thành mảng numpy
    np_arr = np.frombuffer(image_data, np.uint8)
    # Giải mã mảng numpy thành ảnh
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    enhanced_image = enhance_image(image)
    overlayed_image = process_and_overlay(image, enhanced_image)

    if overlayed_image is None:
        return 0, None

    results_model = model(overlayed_image)

    detected_boxes = results_model[0].boxes
    count = 0
    for box in detected_boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if conf < 0.34:
            continue

        count += 1
        cv2.rectangle(overlayed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(overlayed_image, f'Count: {count}, Class: {cls}, Conf: {conf:.2f}',
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Chuyển đổi ảnh sang base64
    base64_image = encode_image_to_base64(overlayed_image)

    return count, base64_image

if __name__ == "__main__":
    model_path = 'training_results_1m.pt' 
    image_base64 = ''

    count, base64_img = count_pipes(image_base64, model_path)
