# 🏗 **Topology của Hệ thống Phát hiện và Đếm Đối tượng**  

## 📁 **1. Thành phần chính**  
- **Input**: Ảnh từ các thư mục theo loại đối tượng (type folder).  
- **Models**:  
  - **YOLOv11m2**: Phát hiện đối tượng trong ảnh.  
  - **Attention U-Net**: Xác định vùng quan tâm (ROI) chứa đối tượng.  
- **Clustering**: Sử dụng **DBSCAN** để xử lý tình huống có nhiều đối tượng gần nhau.  

---

## ⚙️ **2. Quy trình hoạt động**  

```mermaid
flowchart TD
    A[Input Image] --> B[Attention U-Net]
    B --> C[Dự đoán ROI]
    C --> D{ROI Tìm thấy?}
    
    D -->|Không| E[Đếm bằng UNet - Connected Components]
    D -->|Có| F[Crop Image theo ROI]
    
    F --> G[YOLOv11m2 Detection]
    G --> H[NMS - Loại bỏ trùng lặp]
    H --> I{Đối tượng gần nhau?}
    
    I -->|Có| J[DBSCAN Clustering]
    I -->|Không| K[Final YOLO Count]
    
    E --> L[Final UNet Count]
    J --> K
    K --> M[Kết quả cuối cùng]
```

## 🛠 **3. Các hàm chính**

### `pipeline_inference_with_unet_roi(image_path)`
- Chạy **U-Net** để dự đoán vùng chứa vật thể.  
- Chạy **YOLO** trên vùng **ROI** hoặc dùng kết quả **U-Net** nếu **YOLO** không phát hiện.  

### `apply_nms(boxes, iou_threshold)`
- Áp dụng **Non-Max Suppression (NMS)** để loại bỏ các **bbox** trùng lặp.  

### `adjust_for_crowded(predicted_boxes)`
- Dùng thuật toán **DBSCAN** gom nhóm các đối tượng gần nhau.  

### `load_ground_truth_count(label_path, expected_class)`
- Đếm số lượng **ground truth** trong tệp nhãn.  

### `evaluate_yolo_on_type_folder(base_folder)`
- Chạy đánh giá trên từng thư mục theo loại đối tượng.  

---

## 🏷 **4. Kết quả Đầu ra**  

- **YOLO Count**: Số lượng đếm được từ **YOLO**.  
- **UNet Count**: Số lượng đếm được từ **U-Net**.  
- **Final Count**: Kết quả cuối cùng (ưu tiên YOLO nếu có).  
- **ROI BBox**: Vùng bao của **ROI** trong ảnh gốc.  

---

## 📊 **5. Công nghệ và Thư viện**  

- **YOLOv11m2**: Mô hình phát hiện đối tượng.  
- **Attention U-Net**: Phân đoạn ảnh và xác định vùng chứa đối tượng.  
- **DBSCAN (Sklearn)**: Gom nhóm đối tượng.  
- **OpenCV**: Xử lý ảnh.  
- **Numpy, Torch**: Xử lý số liệu và tensor.  
