from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    min,
    max,
    hour,
    to_date,
    year,
    month,
    corr
)

# --------------------------------------------------
# 1. START SPARK
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("Electricity Consumption Analysis") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

file_path = "individual+household+electric+power+consumption/household_power_consumption.txt"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("sep", ";") \
    .option("nullValue", "?") \
    .csv(file_path)


# --------------------------------------------------
# 3. REMOVE MISSING VALUES
# --------------------------------------------------

df_clean = df.na.drop()


# --------------------------------------------------
# 4. CONVERT NUMERIC COLUMNS
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
# 5. OVERALL ELECTRICITY CONSUMPTION
# --------------------------------------------------

consumption_stats = df_clean.select(
    avg("Global_active_power").alias("Average_Consumption"),
    min("Global_active_power").alias("Minimum_Consumption"),
    max("Global_active_power").alias("Maximum_Consumption")
)

print("\n==============================================")
print(" OVERALL ELECTRICITY CONSUMPTION")
print("==============================================")

consumption_stats.show(truncate=False)


# --------------------------------------------------
# 6. HOURLY ELECTRICITY CONSUMPTION
# --------------------------------------------------

df_clean = df_clean.withColumn(
    "Hour",
    hour(col("Time").cast("timestamp"))
)

hourly_consumption = df_clean.groupBy("Hour") \
    .agg(
        avg("Global_active_power").alias("Average_Consumption"),
        max("Global_active_power").alias("Maximum_Consumption")
    ) \
    .orderBy("Hour")

print("\n==============================================")
print(" HOURLY ELECTRICITY CONSUMPTION")
print("==============================================")

hourly_consumption.show(24, truncate=False)


# --------------------------------------------------
# 7. PEAK DEMAND ANALYSIS
# --------------------------------------------------

peak_hour = hourly_consumption.orderBy(
    col("Average_Consumption").desc()
).first()

print("\n==============================================")
print(" PEAK DEMAND")
print("==============================================")

print(
    "Peak Consumption Hour:",
    peak_hour["Hour"]
)

print(
    "Peak Average Consumption:",
    peak_hour["Average_Consumption"],
    "kW"
)

top_peak_hours = hourly_consumption.orderBy(
    col("Average_Consumption").desc()
).limit(5)

print("\nTop 5 Peak Consumption Hours:")

top_peak_hours.show(truncate=False)


# --------------------------------------------------
# 8. DAILY ELECTRICITY CONSUMPTION
# --------------------------------------------------

df_clean = df_clean.withColumn(
    "Day",
    to_date(col("Date"), "d/M/yyyy")
)

daily_consumption = df_clean.groupBy("Day") \
    .agg(
        avg("Global_active_power").alias("Average_Consumption"),
        min("Global_active_power").alias("Minimum_Consumption"),
        max("Global_active_power").alias("Maximum_Consumption")
    ) \
    .orderBy("Day")

print("\n==============================================")
print(" DAILY ELECTRICITY CONSUMPTION")
print("==============================================")

print("\nFirst 10 Days:")

daily_consumption.show(10, truncate=False)


highest_average_day = daily_consumption.orderBy(
    col("Average_Consumption").desc()
).first()

print("\nDay with Highest Average Consumption:")

print(
    "Date:",
    highest_average_day["Day"]
)

print(
    "Average Consumption:",
    highest_average_day["Average_Consumption"],
    "kW"
)


lowest_average_day = daily_consumption.orderBy(
    col("Average_Consumption").asc()
).first()

print("\nDay with Lowest Average Consumption:")

print(
    "Date:",
    lowest_average_day["Day"]
)

print(
    "Average Consumption:",
    lowest_average_day["Average_Consumption"],
    "kW"
)


top_days = daily_consumption.orderBy(
    col("Average_Consumption").desc()
).limit(5)

print("\nTop 5 Days by Average Consumption:")

top_days.show(truncate=False)


# --------------------------------------------------
# 9. MONTHLY ELECTRICITY ANALYSIS
# --------------------------------------------------

monthly_consumption = df_clean.groupBy(
    year("Day").alias("Year"),
    month("Day").alias("Month")
).agg(
    avg("Global_active_power").alias("Average_Consumption"),
    min("Global_active_power").alias("Minimum_Consumption"),
    max("Global_active_power").alias("Maximum_Consumption")
).orderBy(
    "Year",
    "Month"
)

print("\n==============================================")
print(" MONTHLY ELECTRICITY CONSUMPTION")
print("==============================================")

print("\nMonthly Consumption:")

monthly_consumption.show(50, truncate=False)


highest_month = monthly_consumption.orderBy(
    col("Average_Consumption").desc()
).first()

print("\nMonth with Highest Average Consumption:")

print(
    "Year:",
    highest_month["Year"]
)

print(
    "Month:",
    highest_month["Month"]
)

print(
    "Average Consumption:",
    highest_month["Average_Consumption"],
    "kW"
)


lowest_month = monthly_consumption.orderBy(
    col("Average_Consumption").asc()
).first()

print("\nMonth with Lowest Average Consumption:")

print(
    "Year:",
    lowest_month["Year"]
)

print(
    "Month:",
    lowest_month["Month"]
)

print(
    "Average Consumption:",
    lowest_month["Average_Consumption"],
    "kW"
)


# --------------------------------------------------
# 10. YEARLY ELECTRICITY ANALYSIS
# --------------------------------------------------

yearly_consumption = df_clean.groupBy(
    year("Day").alias("Year")
).agg(
    avg("Global_active_power").alias("Average_Consumption"),
    min("Global_active_power").alias("Minimum_Consumption"),
    max("Global_active_power").alias("Maximum_Consumption")
).orderBy("Year")

print("\n==============================================")
print(" YEARLY ELECTRICITY CONSUMPTION")
print("==============================================")

yearly_consumption.show(truncate=False)


highest_year = yearly_consumption.orderBy(
    col("Average_Consumption").desc()
).first()

print("\nYear with Highest Average Consumption:")

print(
    "Year:",
    highest_year["Year"]
)

print(
    "Average Consumption:",
    highest_year["Average_Consumption"],
    "kW"
)


# --------------------------------------------------
# 11. SUB-METER ANALYSIS
# --------------------------------------------------

sub_meter_stats = df_clean.select(
    avg("Sub_metering_1").alias("Sub_meter_1_Average"),
    avg("Sub_metering_2").alias("Sub_meter_2_Average"),
    avg("Sub_metering_3").alias("Sub_meter_3_Average")
)

print("\n==============================================")
print(" SUB-METER ANALYSIS")
print("==============================================")

sub_meter_stats.show(truncate=False)


# --------------------------------------------------
# 12. VOLTAGE / CURRENT / REACTIVE POWER ANALYSIS
# --------------------------------------------------

electrical_stats = df_clean.select(
    avg("Voltage").alias("Average_Voltage"),
    min("Voltage").alias("Minimum_Voltage"),
    max("Voltage").alias("Maximum_Voltage"),

    avg("Global_intensity").alias("Average_Current"),
    min("Global_intensity").alias("Minimum_Current"),
    max("Global_intensity").alias("Maximum_Current"),

    avg("Global_reactive_power").alias("Average_Reactive_Power"),
    min("Global_reactive_power").alias("Minimum_Reactive_Power"),
    max("Global_reactive_power").alias("Maximum_Reactive_Power")
)

print("\n==============================================")
print(" VOLTAGE / CURRENT / REACTIVE POWER")
print("==============================================")

electrical_stats.show(truncate=False)


# --------------------------------------------------
# 13. CORRELATION ANALYSIS
# --------------------------------------------------

print("\n==============================================")
print(" CORRELATION ANALYSIS")
print("==============================================")

active_power_voltage = df_clean.select(
    corr(
        "Global_active_power",
        "Voltage"
    ).alias("ActivePower_Voltage_Correlation")
)

active_power_current = df_clean.select(
    corr(
        "Global_active_power",
        "Global_intensity"
    ).alias("ActivePower_Current_Correlation")
)

active_power_reactive = df_clean.select(
    corr(
        "Global_active_power",
        "Global_reactive_power"
    ).alias("ActivePower_ReactivePower_Correlation")
)

print("\nActive Power vs Voltage:")

active_power_voltage.show(truncate=False)

print("\nActive Power vs Current:")

active_power_current.show(truncate=False)

print("\nActive Power vs Reactive Power:")

active_power_reactive.show(truncate=False)


# --------------------------------------------------
# 14. END SPARK
# --------------------------------------------------

spark.stop()

print("\n==============================================")
print(" ANALYSIS COMPLETED SUCCESSFULLY")
print("==============================================")