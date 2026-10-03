from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Electricity Data Inspection") \
    .master("local[*]") \
    .getOrCreate()

file_path = "individual+household+electric+power+consumption/household_power_consumption.txt"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .csv(file_path)

print("Number of rows:", df.count())

print("\nColumns:")
df.printSchema()

print("\nFirst 5 rows:")
df.show(5, truncate=False)

spark.stop()