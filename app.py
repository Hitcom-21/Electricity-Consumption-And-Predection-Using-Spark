from flask import Flask, render_template, jsonify, request

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    avg,
    max as spark_max,
    min as spark_min,
    hour,
    to_date,
    year,
    month,
    dayofmonth,
    lit
)


app = Flask(__name__)


# ==========================================================
# SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("Electricity Consumption Dashboard") \
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
# OVERALL STATISTICS
# ==========================================================

overall_stats = df_clean.select(
    avg("Global_active_power").alias("Average_Consumption"),
    spark_max("Global_active_power").alias("Maximum_Power")
).first()


average_consumption = round(
    float(overall_stats["Average_Consumption"]),
    4
)


maximum_power = round(
    float(overall_stats["Maximum_Power"]),
    3
)


# ==========================================================
# OVERALL HOURLY ANALYSIS
# ==========================================================

df_with_hour = df_clean.withColumn(
    "Hour",
    hour(col("Time").cast("timestamp"))
)


hourly_overall = df_with_hour.groupBy(
    "Hour"
).agg(
    avg("Global_active_power").alias(
        "Average_Consumption"
    )
).orderBy(
    col("Average_Consumption").desc()
)


peak_row = hourly_overall.first()


peak_hour_number = int(
    peak_row["Hour"]
)


if peak_hour_number == 0:

    peak_hour = "12:00 AM"

elif peak_hour_number < 12:

    peak_hour = f"{peak_hour_number}:00 AM"

elif peak_hour_number == 12:

    peak_hour = "12:00 PM"

else:

    peak_hour = f"{peak_hour_number - 12}:00 PM"


# ==========================================================
# AVAILABLE YEARS
# ==========================================================

available_years_df = df_clean.select(
    year("Day").alias("Year")
).distinct().orderBy("Year")


available_years = [
    int(row["Year"])
    for row in available_years_df.collect()
]


# ==========================================================
# DATE RANGE
# ==========================================================

date_range = df_clean.select(
    spark_min("Day").alias("Min_Date"),
    spark_max("Day").alias("Max_Date")
).first()


min_date = date_range["Min_Date"].strftime(
    "%Y-%m-%d"
)

max_date = date_range["Max_Date"].strftime(
    "%Y-%m-%d"
)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# OVERVIEW API
# ==========================================================

@app.route("/api/overview")
def overview():

    return jsonify({

        "average_consumption":
            average_consumption,

        "peak_hour":
            peak_hour,

        "maximum_power":
            maximum_power,

        "available_years":
            available_years,

        "min_date":
            min_date,

        "max_date":
            max_date

    })


# ==========================================================
# HOURLY API
# ==========================================================

@app.route("/api/hourly")
def hourly_data():

    selected_date = request.args.get(
        "date"
    )


    if not selected_date:

        return jsonify({
            "error": "Date is required"
        }), 400


    try:

        selected_date_value = (
            spark.sql(
                f"""
                SELECT to_date(
                    '{selected_date}'
                ) AS selected_date
                """
            ).first()["selected_date"]
        )

    except Exception:

        return jsonify({
            "error": "Invalid date"
        }), 400


    selected_data = df_clean.filter(
        col("Day") == lit(
            selected_date_value
        )
    )


    hourly = selected_data.withColumn(
        "Hour",
        hour(
            col("Time").cast("timestamp")
        )
    ).groupBy(
        "Hour"
    ).agg(
        avg(
            "Global_active_power"
        ).alias(
            "Average_Consumption"
        )
    ).orderBy(
        "Hour"
    )


    results = hourly.collect()


    hours = [
        int(row["Hour"])
        for row in results
    ]


    values = [
        round(
            float(
                row["Average_Consumption"]
            ),
            3
        )
        for row in results
    ]


    return jsonify({

        "date":
            selected_date,

        "hours":
            hours,

        "values":
            values

    })


# ==========================================================
# MONTHLY API
# ==========================================================

@app.route("/api/monthly")
def monthly_data():

    selected_year = request.args.get(
        "year"
    )


    if not selected_year:

        return jsonify({
            "error": "Year is required"
        }), 400


    try:

        selected_year = int(
            selected_year
        )

    except ValueError:

        return jsonify({
            "error": "Invalid year"
        }), 400


    selected_data = df_clean.filter(
        year("Day") == selected_year
    )


    monthly = selected_data.groupBy(
        year("Day").alias("Year"),
        month("Day").alias("Month")
    ).agg(
        avg(
            "Global_active_power"
        ).alias(
            "Average_Consumption"
        )
    ).orderBy(
        "Month"
    )


    results = monthly.collect()


    months = [
        int(row["Month"])
        for row in results
    ]


    values = [
        round(
            float(
                row["Average_Consumption"]
            ),
            3
        )
        for row in results
    ]


    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]


    labels = [
        month_names[
            month_number - 1
        ]
        for month_number in months
    ]


    return jsonify({

        "year":
            selected_year,

        "months":
            months,

        "labels":
            labels,

        "values":
            values

    })


# ==========================================================
# DAILY API
# ==========================================================

@app.route("/api/daily")
def daily_data():

    selected_year = request.args.get(
        "year"
    )

    selected_month = request.args.get(
        "month"
    )


    if not selected_year or not selected_month:

        return jsonify({
            "error": "Year and month are required"
        }), 400


    try:

        selected_year = int(
            selected_year
        )

        selected_month = int(
            selected_month
        )

    except ValueError:

        return jsonify({
            "error": "Invalid year or month"
        }), 400


    selected_data = df_clean.filter(
        (year("Day") == selected_year) &
        (month("Day") == selected_month)
    )


    daily = selected_data.groupBy(
        dayofmonth("Day").alias("Day")
    ).agg(
        avg(
            "Global_active_power"
        ).alias(
            "Average_Consumption"
        )
    ).orderBy(
        "Day"
    )


    results = daily.collect()


    days = [
        int(row["Day"])
        for row in results
    ]


    values = [
        round(
            float(
                row["Average_Consumption"]
            ),
            3
        )
        for row in results
    ]


    return jsonify({

        "year":
            selected_year,

        "month":
            selected_month,

        "days":
            days,

        "values":
            values

    })


# ==========================================================
# YEARLY API
# ==========================================================

@app.route("/api/yearly")
def yearly_data():

    yearly = df_clean.groupBy(
        year("Day").alias("Year")
    ).agg(
        avg(
            "Global_active_power"
        ).alias(
            "Average_Consumption"
        )
    ).orderBy(
        "Year"
    )


    results = yearly.collect()


    years = [
        int(row["Year"])
        for row in results
    ]


    values = [
        round(
            float(
                row["Average_Consumption"]
            ),
            3
        )
        for row in results
    ]


    return jsonify({

        "years":
            years,

        "values":
            values

    })


# ==========================================================
# START FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )