import base64
import json
import cv2
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from ultralytics import YOLO

#Khởi tạo Spark Streaming
spark = SparkSession.builder \
    .appName("Water Pipe Count") \
    .master("spark://tienlee-virtual-machine:7077") \
    .config("spark.sql.streaming.checkpointLocation", "/tmp/kafka-checkpoints") \
    .getOrCreate()

#Đọc dữ liệu từ Kafka (ảnh base64 từ `kafka_input`)
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "kafka_input") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

#Chuyển đổi dữ liệu Kafka từ binary → string
df = df.selectExpr("CAST(value AS STRING) as message")

#Hàm để phân tích cú pháp JSON an toàn
def safe_json_parse(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

#Hàm YOLO nhận diện ống nước từ ảnh base64
def count_pipe(base64_input):
    try:
        #Load YOLO Model trong mỗi worker (KHÔNG truyền từ driver)
        model = YOLO("./training_results_3m.pt")

        # Giải mã ảnh base64
        image_data = base64.b64decode(base64_input)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Ảnh không hợp lệ hoặc không thể giải mã.")

        # Chạy YOLO model
        results = model(image)
        detected_boxes = results[0].boxes
        total_pipe = len(detected_boxes)

        # Mã hóa ảnh kết quả
        retval, buffer = cv2.imencode('.jpg', image)
        out_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pipe": total_pipe,
            "out_image": out_image_base64
        }

    except Exception as e:
        return {
            "status": "error",
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_message": str(e)
        }

#Xử lý từng batch ảnh từ Kafka
def process_batch(batch_df, batch_id):
    try:
        # Kiểm tra nếu batch rỗng
        if batch_df.rdd.isEmpty():
            print(f"⚠️ Batch {batch_id} không có dữ liệu, bỏ qua...")
            return

        #Chuyển đổi RDD
        processed_rdd = (
            batch_df.rdd
            .map(lambda row: safe_json_parse(row["message"]))  # Xử lý JSON an toàn
            .filter(lambda data: data is not None and "image" in data)  # Lọc dữ liệu hợp lệ
            .map(lambda data: count_pipe(data["image"]))  # Xử lý ảnh
        )

        if processed_rdd.isEmpty():
            print(f"⚠️ Batch {batch_id} không có kết quả hợp lệ, bỏ qua...")
            return

        #Xử lý kết quả thành công
        success_rdd = processed_rdd.filter(lambda res: res["status"] == "success")
        if not success_rdd.isEmpty():
            formatted_rdd = success_rdd.map(lambda x: (json.dumps(x),))  # Định dạng tuple (value,)
            success_df = spark.createDataFrame(formatted_rdd, ["value"])
            
            if not success_df.rdd.isEmpty():  # Đảm bảo có dữ liệu trước khi ghi vào Kafka
                success_df.write \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", "localhost:9092") \
                    .option("topic", "kafka_output") \
                    .save()

        #Xử lý lỗi
        error_rdd = processed_rdd.filter(lambda res: res["status"] == "error")
        if not error_rdd.isEmpty():
            error_df = spark.createDataFrame(error_rdd.map(lambda x: (json.dumps(x),)), ["value"])
            
            if not error_df.rdd.isEmpty():  # Kiểm tra trước khi ghi vào Kafka
                error_df.write \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", "localhost:9092") \
                    .option("topic", "kafka_output") \
                    .save()

        print(f"✅ Batch {batch_id} xử lý xong.")

    except Exception as e:
        print(f"❌ Lỗi batch {batch_id}: {e}")



#Ghi dữ liệu đã xử lý vào Kafka (`kafka_output`)
df.writeStream \
    .foreachBatch(lambda batch_df, batch_id: process_batch(batch_df, batch_id)) \
    .start()

#Chạy Spark Streaming
spark.streams.awaitAnyTermination()