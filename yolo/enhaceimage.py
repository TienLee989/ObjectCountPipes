import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm  # Thêm tqdm để có thanh tiến trình


def enhance_image(image):
    """
    Áp dụng các bước xử lý ảnh lên một ảnh.

    Args:
        image: Ảnh đầu vào (NumPy array).

    Returns:
        Ảnh đã được xử lý (NumPy array).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Phép toán điểm (điều chỉnh)
    alpha = 1.2
    beta = 10
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 5. Cân bằng histogram thích ứng (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_eq = clahe.apply(adjusted)

    # 6. Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(adaptive_eq, (3, 3), 0)

    # 7. Unsharp Masking (trước Sobel)
    blurred = cv2.GaussianBlur(gaussian_blur, (5, 5), 0)
    unsharp_mask = cv2.addWeighted(gaussian_blur, 1.2, blurred, -0.2, 0)

    # 8. Sobel
    sobelx = cv2.Sobel(unsharp_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(unsharp_mask, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs_x = cv2.convertScaleAbs(sobelx)
    sobel_abs_y = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

    # 9. Dilation (tùy chọn, có thể bỏ qua hoặc điều chỉnh kernel)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(sobel_combined, kernel, iterations=1)

    return dilation  # Hoặc trả về ảnh ở bước bạn muốn (ví dụ: unsharp_mask, sobel_combined)


def process_dataset(base_dir, output_dir):
    """
    Xử lý toàn bộ dataset, tạo thư mục mới và lưu ảnh đã xử lý.

    Args:
        base_dir: Đường dẫn đến thư mục gốc của dataset (chứa train, valid, test).
        output_dir: Đường dẫn đến thư mục đầu ra (dataset_enhance).
    """

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:  # Duyệt qua các tập train, valid, test
        image_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        output_image_dir = os.path.join(output_dir, split, "images")
        output_label_dir = os.path.join(output_dir, split, "labels")

        # Tạo thư mục con (images, labels) trong thư mục đầu ra
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # Lấy danh sách file ảnh
        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))]  # Thêm các đuôi file ảnh khác nếu cần

        # Duyệt qua từng file ảnh và xử lý
        for image_file in tqdm(image_files, desc=f"Processing {split} images"):
            image_path = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # Xử lý ảnh
            enhanced_img = enhance_image(img)

            # Lưu ảnh đã xử lý
            output_image_path = os.path.join(output_image_dir, image_file)
            cv2.imwrite(output_image_path, enhanced_img)

            # Copy file label
            label_file = os.path.splitext(image_file)[0] + ".txt"  # Tên file label tương ứng
            label_path = os.path.join(label_dir, label_file)
            output_label_path = os.path.join(output_label_dir, label_file)

            if os.path.exists(label_path):  # Kiểm tra xem file label có tồn tại không
                shutil.copyfile(label_path, output_label_path)  # Copy file label
            else:
                print(f"Warning: Label file not found for {image_file}. Skipping label copy.")



# Đường dẫn đến thư mục gốc của dataset (chứa train, valid, test)
base_dataset_dir = "."  # Thay "." bằng đường dẫn thực tế nếu thư mục hiện tại không phải là thư mục chứa train, valid, test

# Đường dẫn đến thư mục đầu ra (dataset_enhance)
output_dataset_dir = "dataset_enhance"

# Thực hiện xử lý
process_dataset(base_dataset_dir, output_dataset_dir)
print("Done!")