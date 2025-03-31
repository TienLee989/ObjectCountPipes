# 🏗 **Topology của Hệ thống Phát hiện và Đếm Đối tượng**  

##### Link model training
[Model yolo after 700 epochs and enhance image](https://drive.google.com/file/d/1zwTjVbIii7afvw-E-_x_N18gXeMUzjfq/view?usp=drive_link)

[Attention Unet5](https://drive.google.com/file/d/1Sg7u8kZa7zPE3XgtB3xg-Ed3MitKNlBr/view?usp=drive_link)

[Những hình ảnh đầu ra cảu quá trình test](https://drive.google.com/drive/folders/1cb-z6BdstMr9oSzAnVAikZuJ4JVnyvZ9?usp=drive_link)

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

## 🏗 Topology cho mô hình Attention U-Net và quy trình huấn luyện
Huấn luyện mô hình Attention U-Net với Self-Attention cho bài toán phân đoạn ảnh (Image Segmentation)

```mermaid
flowchart TD
  subgraph "Xu ly Dataset"
    A[Load Annotations from JSON file]
    B[Mapping image_id to file_name]
    C[Mapping file_name to annotations]
    D[Data Generator]
    E[Load Image from image_dir]
    F[Create Mask from bbox - Annotation]
    G[Resize Image and Mask to input_size]
    H[Output Batch: Images and Masks]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
  end
```

```mermaid
flowchart TD
    subgraph "Xay dung Attention U-Net"
      I[Input Layer: input_shape]
      J[Encoder Block 1: Conv Block 64 + MaxPooling]
      K[Encoder Block 2: Conv Block 128 + MaxPooling]
      L[Encoder Block 3: Conv Block 256 + MaxPooling]
      M[Encoder Block 4: Conv Block 512 + MaxPooling]
      N[Encoder Block 5: Conv Block 1024]
      O[Self-Attention Layer: Bottleneck]
      P[Decoder Block 1: UpSampling + AttentionGate skip Block4 + Concatenate + Conv Block 512]
      Q[Decoder Block 2: UpSampling + AttentionGate skip Block3 + Concatenate + Conv Block 256]
      R[Decoder Block 3: UpSampling + AttentionGate skip Block2 + Concatenate + Conv Block 128]
      S[Decoder Block 4: UpSampling + AttentionGate skip Block1 + Concatenate + Conv Block 64]
      T[Output Layer: Conv2D with softmax]
      I --> J
      J --> K
      K --> L
      L --> M
      M --> N
      N --> O
      O --> P
      P --> Q
      Q --> R
      R --> S
      S --> T
    end

    subgraph "Huan luyen mo hinh"
      U[Train Generator: batch_size, epochs]
      V[Compile Model: optimizer Adam, loss sparse_categorical_crossentropy]
      W[Model.fit: Train, Validation, Callbacks]
      X[ModelCheckpoint Callback: Save best weights]
      Y[Save final weights: attention_unet_coco.h5]
      T --> U
      U --> V
      V --> W
      W --> X
      X --> Y
    end
```
