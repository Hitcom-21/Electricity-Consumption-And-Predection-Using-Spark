from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, to_timestamp

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Electricity Data Cleaning") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()


# --------------------------------------------------
# 2. Dataset path
# --------------------------------------------------

file_path = "individual+household+electric+power+consumption/household_power_consumption.txt"


# --------------------------------------------------
# 3. Load the raw dataset
# --------------------------------------------------

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("sep", ";") \
    .option("nullValue", "?") \
    .csv(file_path)

print("Original rows:", df.count())


# --------------------------------------------------
# 4. Remove rows containing missing values
# --------------------------------------------------

df_clean = df.na.drop()

print("Rows after removing missing values:", df_clean.count())


# --------------------------------------------------
# 5. Convert numerical columns to double
# --------------------------------------------------

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in numeric_columns:
    df_clean = df_clean.withColumn(
        column,
        col(column).cast("double")
    )


# --------------------------------------------------
# 6. Create proper DateTime column
# --------------------------------------------------

df_clean = df_clean.withColumn(
    "DateTime",
    to_timestamp(
        concat_ws(" ", col("Date"), col("Time")),
        "dd/MM/yyyy HH:mm:ss"
    )
)


# --------------------------------------------------
# 7. Display cleaned schema
# --------------------------------------------------

print("\nCleaned schema:")
df_clean.printSchema()


# --------------------------------------------------
# 8. Display first 5 cleaned records
# --------------------------------------------------

print("\nFirst 5 cleaned rows:")
df_clean.show(5, truncate=False)


# --------------------------------------------------
# 9. Stop Spark
# --------------------------------------------------

spark.stop()