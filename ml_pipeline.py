import os
import numpy as np
import cv2
import time
import threading
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, udf, lit
from pyspark.sql.types import StructType, StringType, DoubleType, FloatType

# --- WINDOWS COMPATIBILITY ---
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 pyspark-shell"

# Shared memory for Video Risk
VIDEO_SCORE = {"value": 0.0}

# 1. VIDEO THREAD (CCTV)
def run_video():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 0 is default webcam. Replace with URL string if using IP Camera
    cap = cv2.VideoCapture(0) 
    
    print("[VIDEO] Camera feed started...")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
        
        frame = cv2.resize(frame, (640, 480))
        rects, _ = hog.detectMultiScale(frame, winStride=(8,8))
        
        # Risk: 20 points per person detected, max 100
        VIDEO_SCORE["value"] = float(min(len(rects) * 20.0, 100.0))
        
        # Show feed (Optional - remove for headless)
        cv2.putText(frame, f"Physical Risk: {VIDEO_SCORE['value']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Sentinel CCTV Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

threading.Thread(target=run_video, daemon=True).start()

# 2. SPARK FUSION ENGINE
spark = SparkSession.builder \
    .appName("Sentinel-Fusion-Core") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define Schema for Kafka JSON
schema = StructType() \
    .add("timestamp", StringType()) \
    .add("indicator", StringType()) \
    .add("type", StringType()) \
    .add("confidence", DoubleType())

# UDFs for ML Logic
def calculate_url_risk(conf, threat_type):
    base = conf * 50
    if threat_type == "malware_delivery": base += 40
    if threat_type == "phishing": base += 20
    return float(min(base, 100.0))

url_risk_udf = udf(calculate_url_risk, FloatType())
video_risk_udf = udf(lambda: float(VIDEO_SCORE["value"]), FloatType())

# Kafka Source
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "sentinel-raw-events") \
    .option("startingOffsets", "latest") \
    .load()

parsed = raw_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Fusion Logic (60% URL, 40% Video)
fused = parsed.withColumn("URL_SCORE", url_risk_udf(col("confidence"), col("type"))) \
              .withColumn("VIDEO_SCORE", video_risk_udf()) \
              .withColumn("FUSED_SCORE", (col("URL_SCORE") * 0.6) + (video_risk_udf() * 0.4))

# Write to Console AND to a JSON file for the RL Layer
# Note: Checkpoint is critical for Windows 11
# --- UPDATED SINK FOR ml_pipeline.py ---
query = fused.writeStream \
    .outputMode("append") \
    .format("json") \
    .option("path", "C:/tmp/sentinel_results") \
    .option("checkpointLocation", "C:/tmp/spark_fusion_checkpoints") \
    .start()

print("[SYSTEM] Fusion Engine is now exporting real-time data to C:/tmp/sentinel_results")
query.awaitTermination()