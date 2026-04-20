from pyspark.sql import SparkSession
import logging

class SparkManager:
    def __init__(self):
        self.spark = None

    def get_session(self):
        if self.spark is None:
            try:
                self.spark = SparkSession.builder \
                    .appName("MedTex-Batch-Processor") \
                    .config("spark.driver.memory", "2g") \
                    .getOrCreate()
            except Exception as e:
                logging.error(f"Spark failed to start: {e}")
        return self.spark

spark_manager = SparkManager()