import os
import time
import json
import requests
import threading
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, DoubleType

# --- 1. THE LIVE URL FETCHING LOGIC (The "Producer" inside Spark) ---
def fetch_live_threats():
    """Fetches live malicious URLs from an open threat feed."""
    producer = KafkaProducer(
        bootstrap_servers=['localhost:29092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Example URL: OpenPhish (Live Phishing Feed)
    # Note: Replace with your specific bank/cyber URL if different
    THREAT_URL = "https://openphish.com/feed.txt" 
    
    print(f"[FETCH] Starting background thread for URL: {THREAT_URL}")
    
    while True:
        try:
            response = requests.get(THREAT_URL, timeout=10)
            if response.status_code == 200:
                # Get the first 3 new threats to avoid flooding
                lines = response.text.splitlines()[:3]
                for url in lines:
                    data = {
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "indicator": url,
                        "type": "phishing",
                        "confidence": 0.85  # Default confidence for this feed
                    }
                    producer.send('sentinel-raw-events', value=data)
                
                print(f"[FETCH] Successfully pushed {len(lines)} threats to Kafka.")
        except Exception as e:
            print(f"[FETCH ERROR] {e}")
        
        time.sleep(30)  # Wait 30 seconds before checking the URL again

# Start the URL fetcher in the background
threading.Thread(target=fetch_live_threats, daemon=True).start()


# --- 2. THE SPARK KAFKA INGESTION ENGINE ---
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 pyspark-shell"

spark = SparkSession.builder \
    .appName("Sentinel-Ingestion") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define schema to match the URL data
schema = StructType() \
    .add("timestamp", StringType()) \
    .add("indicator", StringType()) \
    .add("type", StringType()) \
    .add("confidence", DoubleType())

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "sentinel-raw-events") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON data
parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Add a simple Risk Level calculation
enriched_df = parsed_df.withColumn("risk_level", 
    col("confidence") * 100)

print(">>> Sentinel Ingestion Engine LIVE (Fetching from URL)...")

# Write to console
query = enriched_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("checkpointLocation", "C:/tmp/spark_kafka_checkpoints") \
    .start()

query.awaitTermination()