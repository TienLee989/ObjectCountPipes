import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

def process_and_overlay(original_image_path, enhanced_image_path, output_path, white_threshold=70, lower_white_threshold=50):
    """
    Xử lý ảnh enhance, tạo viền đen, và đè lên ảnh gốc.

    Args:
        original_image_path: Đường dẫn ảnh gốc.
        enhanced_image_path: Đường dẫn ảnh đã enhance.
        output_path: Đường dẫn lưu kết quả.
        white_threshold: Ngưỡng trên để xác định màu trắng (>= ngưỡng này là trắng).
        lower_white_threshold: Ngưỡng dưới (loại bỏ màu trắng < ngưỡng này).
    """

    try:
        # --- Bước 1: Xử lý ảnh đã enhance ---
        enhanced_img = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)
        if enhanced_img is None:
            print(f"Error: Could not read enhanced image at {enhanced_image_path}")
            return
        # Tạo mask: Giữ lại vùng trắng >= white_threshold
        _, mask = cv2.threshold(enhanced_img, white_threshold, 255, cv2.THRESH_BINARY)

        #Loại bỏ vùng trắng < lower_white_threshold.
        _, lower_white_mask = cv2.threshold(enhanced_img, lower_white_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_white_mask))

        # Chuyển thành ảnh RGBA với alpha channel (nền trong suốt)
        rgba_image = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGRA)
        rgba_image[:, :, 3] = mask  # Kênh alpha là mask


        # --- Bước 2: Tạo viền đen ---
        # Chuyển vùng trắng thành đen
        rgba_image[mask > 0, 0:3] = [0, 0, 0]  # BGR = [0, 0, 0] là màu đen

        # Tạo viền đen (dilation trên kênh alpha)
        kernel = np.ones((3, 3), np.uint8)  # Kernel 3x3 (tạo viền 1px xung quanh)
        dilated_alpha = cv2.dilate(rgba_image[:, :, 3], kernel, iterations=1)
        rgba_image[:, :, 3] = dilated_alpha # Gán lại kênh alpha

        # --- Bước 3: Đè lên ảnh gốc ---
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            print(f"Error: Could not read original image at {original_image_path}")
            return

        # Resize PNG to match original image size
        rgba_image = cv2.resize(rgba_image, (original_img.shape[1], original_img.shape[0]))

        # Lấy phần alpha (mask) của ảnh PNG
        alpha_mask = rgba_image[:, :, 3] / 255.0  # Chuẩn hóa về [0, 1]
        alpha_mask_3channel = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)  # Mở rộng thành 3 kênh

        # Tính toán kết quả
        foreground = (rgba_image[:, :, :3] * alpha_mask_3channel).astype(np.uint8)  # Phần overlay (viền đen)
        background = (original_img * (1 - alpha_mask_3channel)).astype(np.uint8)  # Phần ảnh gốc
        final_result = cv2.add(foreground, background)
        cv2.imwrite(output_path, final_result)

    except FileNotFoundError:
        print(f"Error: Image file not found: {original_image_path} or {enhanced_image_path}")
    except Exception as e:
        print(f"Error processing {original_image_path}: {e}")

def process_images(original_base_dir, enhanced_base_dir, output_base_dir):
    """Xử lý toàn bộ thư mục, bao gồm train, valid, test và copy label."""

    for split in ["train", "valid", "test"]:
        original_image_dir = os.path.join(original_base_dir, split, "images")
        enhanced_image_dir = os.path.join(enhanced_base_dir, split, "images")
        output_image_dir = os.path.join(output_base_dir, split, "images")
        output_label_dir = os.path.join(output_base_dir, split, "labels") # Thư mục label đầu ra

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)  # Tạo thư mục label


        for filename in tqdm(os.listdir(enhanced_image_dir), desc=f"Processing {split} images"):
            if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                base_filename = os.path.splitext(filename)[0]  # Tên file không có đuôi
                enhanced_path = os.path.join(enhanced_image_dir, filename)
                original_path = os.path.join(original_image_dir, filename)
                output_path = os.path.join(output_image_dir, filename)

                # Xử lý ảnh và tạo overlay
                process_and_overlay(original_path, enhanced_path, output_path)

                # Copy label
                label_filename = base_filename + ".txt"
                original_label_path = os.path.join(original_base_dir, split, "labels", label_filename)
                output_label_path = os.path.join(output_label_dir, label_filename)

                try:
                    shutil.copyfile(original_label_path, output_label_path)
                except FileNotFoundError:
                    print(f"Warning: Label file not found for {filename} in {split}. Skipping label copy.")
                except Exception as e:
                    print(f"Error copying label for {filename}: {e}")



# Đường dẫn đến thư mục gốc của dataset gốc (chứa train, valid, test)
original_base_dir = "./datasets-origin"

# Đường dẫn đến thư mục gốc của dataset đã enhance (chứa train, valid, test)
enhanced_base_dir = "./dataset_enhance"

# Đường dẫn đến thư mục đầu ra (overlay_output)
output_base_dir = "./overlay_output"


process_images(original_base_dir, enhanced_base_dir, output_base_dir)

print("Done!")