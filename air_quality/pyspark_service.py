# airqualityapp/pyspark_service.py
from pyspark.sql import SparkSession

def perform_air_quality_analysis(csv_path):
    # Créer une session Spark
    spark = SparkSession.builder.appName("air_quality").getOrCreate()

    # Charger les données depuis le fichier CSV
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Effectuer des opérations de transformation ou d'analyse
    result_df = df.select("timestamp", "pollutant1", "pollutant2")

    # Convertir le résultat en Pandas DataFrame (facultatif)
    result_pandas_df = result_df.toPandas()

    return result_pandas_df
