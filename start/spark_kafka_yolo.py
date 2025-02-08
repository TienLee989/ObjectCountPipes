import json
import base64
import cv2
import numpy as np
from pyspark.sql import SparkSession
from ultralytics import YOLO

#Khởi tạo Spark Streaming
spark = SparkSession.builder \
    .appName("Test Kafka Input & YOLO Pipe Counting") \
    .master("spark://tienlee-virtual-machine:7077") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "kafka_input") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load() \
    .selectExpr("CAST(value AS STRING) as message")  # Chuyển message Kafka thành string

#Load YOLO Model trong mỗi worker (KHÔNG truyền từ driver)
model = YOLO("./training_results_3m.pt")

#Hàm YOLO nhận diện ống nước từ ảnh base64
def count_pipe(base64_input):
    try:
        # Giải mã ảnh base64
        image_data = base64.b64decode(base64_input)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Ảnh không hợp lệ hoặc không thể giải mã.")

        # Chạy YOLO model
        results = model(image)
        total_pipe = len(results[0].boxes)

        return total_pipe  # Chỉ trả về số lượng ống nước

    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        return None  # Nếu lỗi thì trả về None

#Xử lý batch dữ liệu từ Kafka
def process_batch(batch_df, batch_id):
    data_list = batch_df.select("message").toPandas()["message"].tolist()  # Chuyển thành danh sách Python

    #Xử lý từng ảnh nhận được từ Kafka
    for data in data_list:
        try:
            json_data = json.loads(data)  # Parse JSON từ message
            if "image" in json_data:  # Kiểm tra có dữ liệu ảnh không
                total_pipes = count_pipe(json_data["image"])  # Xử lý ảnh với YOLO
                print(f"✅ Batch {batch_id} - Ống nước đếm được: {total_pipes}")
            else:
                print(f"⚠️ Batch {batch_id} - Không có dữ liệu ảnh hợp lệ!")
        except json.JSONDecodeError:
            print(f"❌ Batch {batch_id} - JSON không hợp lệ!")

#Bắt đầu Spark Streaming
df.writeStream \
    .foreachBatch(process_batch) \
    .start()

spark.streams.awaitAnyTermination()

