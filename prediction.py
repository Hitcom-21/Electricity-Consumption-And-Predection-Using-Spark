from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    to_date,
    year,
    month,
    dayofmonth,
    dayofweek,
    lag,
    lit,
    min,
    max
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from datetime import date
import csv

# ==========================================================
# SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("Electricity Demand Prediction") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ==========================================================
# LOAD DATASET
# ==========================================================

file_path = (
    "individual+household+electric+power+consumption/"
    "household_power_consumption.txt"
)

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("sep", ";") \
    .option("nullValue", "?") \
    .csv(file_path)


# ==========================================================
# REMOVE MISSING VALUES
# ==========================================================

df_clean = df.na.drop()


# ==========================================================
# CONVERT NUMERIC COLUMNS
# ==========================================================

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


# ==========================================================
# CREATE DATE COLUMN
# ==========================================================

df_clean = df_clean.withColumn(
    "Day",
    to_date(
        col("Date"),
        "d/M/yyyy"
    )
)


# ==========================================================
# CREATE DAILY DATASET
# ==========================================================

daily_data = df_clean.groupBy(
    "Day"
).agg(
    avg("Global_active_power").alias(
        "Daily_Average_Power"
    )
).orderBy(
    "Day"
)


# ==========================================================
# CREATE TIME FEATURES
# ==========================================================

daily_data = daily_data.withColumn(
    "Year",
    year("Day")
)

daily_data = daily_data.withColumn(
    "Month",
    month("Day")
)

daily_data = daily_data.withColumn(
    "Day_of_Month",
    dayofmonth("Day")
)

daily_data = daily_data.withColumn(
    "Day_of_Week",
    dayofweek("Day")
)


# ==========================================================
# CREATE PREVIOUS DAY FEATURES
# ==========================================================

window_spec = Window.orderBy("Day")

daily_data = daily_data.withColumn(
    "Previous_Day_Power",
    lag("Daily_Average_Power", 1).over(window_spec)
)

daily_data = daily_data.withColumn(
    "Previous_2_Day_Power",
    lag("Daily_Average_Power", 2).over(window_spec)
)

daily_data = daily_data.withColumn(
    "Previous_3_Day_Power",
    lag("Daily_Average_Power", 3).over(window_spec)
)


# ==========================================================
# REMOVE ROWS WITHOUT PREVIOUS-DAY VALUES
# ==========================================================

prediction_data = daily_data.na.drop()


# ==========================================================
# DISPLAY PREDICTION DATASET
# ==========================================================

print("\nPrediction Dataset Schema:")
prediction_data.printSchema()

print("\nNumber of prediction records:")
print(prediction_data.count())

print("\nFirst 10 prediction records:")
prediction_data.show(
    10,
    truncate=False
)


# =========================================================python prediction.py=
# TRAIN / TEST SPLIT
# ==========================================================

# Training data: up to December 2021
train_data = prediction_data.filter(
    col("Day") <= lit(date(2021, 12, 31))
)

# Testing data: January 2022 onwards
test_data = prediction_data.filter(
    col("Day") >= lit(date(2022, 1, 1))
)


# ==========================================================
# DISPLAY TRAINING DATA INFORMATION
# ==========================================================

print("\nTraining Data:")
print("Number of records:", train_data.count())

train_data.select(
    min("Day").alias("Start_Date"),
    max("Day").alias("End_Date")
).show()


# ==========================================================
# DISPLAY TESTING DATA INFORMATION
# ==========================================================

print("\nTesting Data:")
print("Number of records:", test_data.count())

test_data.select(
    min("Day").alias("Start_Date"),
    max("Day").alias("End_Date")
).show()


# ==========================================================
# SELECT FEATURES AND TARGET
# ==========================================================

feature_columns = [
    "Previous_Day_Power",
    "Previous_2_Day_Power",
    "Previous_3_Day_Power",
    "Year",
    "Month",
    "Day_of_Month",
    "Day_of_Week"
]

target_column = "Daily_Average_Power"


print("\nFeatures used for prediction:")

for feature in feature_columns:
    print("-", feature)


print("\nTarget:")
print("-", target_column)


# ==========================================================
# PREPARE FEATURES FOR SPARK MLlib
# ==========================================================




# Create feature assembler
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)


# Apply assembler to training data
train_ml = assembler.transform(train_data)


# Apply assembler to testing data
test_ml = assembler.transform(test_data)


# Rename target column as label
train_ml = train_ml.withColumnRenamed(
    target_column,
    "label"
)

test_ml = test_ml.withColumnRenamed(
    target_column,
    "label"
)


# ==========================================================
# DISPLAY ML DATA
# ==========================================================

print("\nTraining data prepared for ML:")
train_ml.select(
    "Day",
    "features",
    "label"
).show(5, truncate=False)


print("\nTesting data prepared for ML:")
test_ml.select(
    "Day",
    "features",
    "label"
).show(5, truncate=False)


# ==========================================================
# TRAIN REGRESSION MODELS
# ==========================================================

from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)


# ==========================================================
# LINEAR REGRESSION
# ==========================================================

print("\nTraining Linear Regression model...")

linear_model = LinearRegression(
    featuresCol="features",
    labelCol="label"
)

linear_model = linear_model.fit(train_ml)

linear_predictions = linear_model.transform(test_ml)

print("Linear Regression completed.")


# ==========================================================
# RANDOM FOREST REGRESSION
# ==========================================================

print("\nTraining Random Forest Regression model...")

rf_model = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    seed=42
)

rf_model = rf_model.fit(train_ml)

rf_predictions = rf_model.transform(test_ml)

print("Random Forest Regression completed.")


# ==========================================================
# GRADIENT BOOSTED TREE REGRESSION
# ==========================================================

print("\nTraining GBT Regression model...")

gbt_model = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    seed=42
)

gbt_model = gbt_model.fit(train_ml)

gbt_predictions = gbt_model.transform(test_ml)

print("GBT Regression completed.")


# ==========================================================
# DISPLAY SAMPLE PREDICTIONS
# ==========================================================

print("\nLinear Regression Predictions:")
linear_predictions.select(
    "Day",
    "label",
    "prediction"
).show(10, truncate=False)


print("\nRandom Forest Predictions:")
rf_predictions.select(
    "Day",
    "label",
    "prediction"
).show(10, truncate=False)


print("\nGBT Predictions:")
gbt_predictions.select(
    "Day",
    "label",
    "prediction"
).show(10, truncate=False)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

from pyspark.ml.evaluation import RegressionEvaluator

# Create evaluators
mae_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

rmse_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

# ----------------------------------------------------------
# Linear Regression Evaluation
# ----------------------------------------------------------

linear_mae = mae_evaluator.evaluate(linear_predictions)
linear_rmse = rmse_evaluator.evaluate(linear_predictions)
linear_r2 = r2_evaluator.evaluate(linear_predictions)

# ----------------------------------------------------------
# Random Forest Evaluation
# ----------------------------------------------------------

rf_mae = mae_evaluator.evaluate(rf_predictions)
rf_rmse = rmse_evaluator.evaluate(rf_predictions)
rf_r2 = r2_evaluator.evaluate(rf_predictions)

# ----------------------------------------------------------
# GBT Evaluation
# ----------------------------------------------------------

gbt_mae = mae_evaluator.evaluate(gbt_predictions)
gbt_rmse = rmse_evaluator.evaluate(gbt_predictions)
gbt_r2 = r2_evaluator.evaluate(gbt_predictions)

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

print("\nLinear Regression:")
print("MAE :", linear_mae)
print("RMSE:", linear_rmse)
print("R2  :", linear_r2)

print("\nRandom Forest Regression:")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)

print("\nGBT Regression:")
print("MAE :", gbt_mae)
print("RMSE:", gbt_rmse)
print("R2  :", gbt_r2)

print("\n" + "=" * 70)

# ==========================================================
# SAVE PREDICTION RESULTS
# ==========================================================

prediction_results = (
    linear_predictions
    .select(
        col("Day"),
        col("label").alias("Actual"),
        col("prediction").alias("Linear_Regression")
    )
    .join(
        rf_predictions.select(
            "Day",
            col("prediction").alias("Random_Forest")
        ),
        on="Day"
    )
    .join(
        gbt_predictions.select(
            "Day",
            col("prediction").alias("GBT")
        ),
        on="Day"
    )
    .orderBy("Day")
)

prediction_rows = prediction_results.collect()

with open("prediction_results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Day",
        "Actual",
        "Linear_Regression",
        "Random_Forest",
        "GBT"
    ])

    for row in prediction_rows:
        writer.writerow([
            str(row["Day"]),
            row["Actual"],
            row["Linear_Regression"],
            row["Random_Forest"],
            row["GBT"]
        ])

# ==========================================================
# SAVE MODEL METRICS
# ==========================================================

with open("model_metrics.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ])

    writer.writerow([
        "Linear Regression",
        linear_mae,
        linear_rmse,
        linear_r2
    ])

    writer.writerow([
        "Random Forest",
        rf_mae,
        rf_rmse,
        rf_r2
    ])

    writer.writerow([
        "GBT Regression",
        gbt_mae,
        gbt_rmse,
        gbt_r2
    ])

print("\nPrediction results saved to: prediction_results.csv")
print("Model metrics saved to: model_metrics.csv")

print("\n" + "=" * 70)
print("PREDICTION FILE GENERATION COMPLETED")
print("=" * 70)

spark.stop()