import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
import spacy

# Ensure the backend folder is in the python path so we can import our engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.app.core.config import settings

class Med7BatchProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MedTex-Batch-NER") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        # Register UDF after SparkSession is created
        self.extract_udf = self._register_udf()

    def _register_udf(self):
        """Register pandas UDF - must be called after SparkSession exists"""
        @pandas_udf("array<struct<text:string,label:string,start:int,end:int>>")
        def extract_med7_entities(texts: pd.Series) -> pd.Series:
            """
            This runs on Spark Workers. 
            We load the model inside the UDF so it's loaded once per worker process.
            """
            # Lazy loading the model on the worker
            nlp = spacy.load(settings.MODEL_NAME)
            
            results = []
            for doc in nlp.pipe(texts.astype(str), batch_size=50):
                entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ]
                results.append(entities)
            
            return pd.Series(results)
        
        return extract_med7_entities

    def run(self, input_path: str, output_path: str, text_column: str):
        """
        Main execution logic for the Spark Job
        """
        print(f"--- Starting MedTex Batch Job: {input_path} ---")

        # 1. Load Data (Supports CSV, Parquet, or JSON)
        if input_path.endswith(".csv"):
            df = self.spark.read.option("header", "true").csv(input_path)
        else:
            df = self.spark.read.parquet(input_path)

        # 2. Apply the NER Transformation
        # We create a new column 'med7_entities' containing the structured JSON
        df_result = df.withColumn(
            "med7_entities", 
            self.extract_udf(df[text_column])
        )

        # 3. Save the results
        # Parquet is preferred for clinical data as it preserves the schema
        df_result.write.mode("overwrite").parquet(output_path)
        
        print(f"--- Job Complete. Results saved to: {output_path} ---")
        self.spark.stop()


@staticmethod
@pandas_udf("array<struct<text:string,label:string,start:int,end:int>>")
def extract_medtex_entities(texts: pd.Series) -> pd.Series:
    # Load both models on the worker
    nlp_med7 = spacy.load("en_core_med7_lg")
    nlp_sci = spacy.load("en_ner_bionlp13cg_md")
    
    results = []
    for text in texts:
        # Get entities from Med7
        doc_med7 = nlp_med7(text)
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc_med7.ents
        ]
        
        # Get entities from SciSpacy (filter for ANATOMY, CHEMICAL, ORGANISM)
        doc_sci = nlp_sci(text)
        sci_entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc_sci.ents if ent.label_ in ["ANATOMY", "CHEMICAL", "ORGANISM"]
        ]
        
        # Merge: avoid duplicates by checking start position
        for sci_ent in sci_entities:
            if not any(e['start'] == sci_ent['start'] for e in entities):
                entities.append(sci_ent)
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        results.append(entities)
    
    return pd.Series(results)


if __name__ == "__main__":
    # Usage: python batch_processor.py <input_file> <output_dir> <column_name>
    # Example: python batch_processor.py data/notes.csv data/output clinical_note
    if len(sys.argv) < 4:
        print("Usage: python batch_processor.py <input_path> <output_path> <text_column>")
    else:
        processor = Med7BatchProcessor()
        processor.run(sys.argv[1], sys.argv[2], sys.argv[3])