# Electricity Consumption Analysis and Energy Demand Prediction Using Apache Spark

## 📌 Project Overview

This project focuses on analyzing household electricity consumption data and predicting future energy demand using **Apache Spark** and **PySpark**.

The project uses the **Individual Household Electric Power Consumption** dataset, which contains electricity usage measurements collected at a household level. The data is processed, cleaned, analyzed, and used for building a machine learning model for energy demand prediction.

The main goal of the project is to demonstrate how **Big Data technologies and machine learning** can be used together to understand electricity consumption patterns and predict energy demand.

---

## 🎯 Objectives

* Analyze household electricity consumption patterns.
* Process and clean large-scale electricity consumption data using Apache Spark.
* Perform exploratory data analysis on electricity usage.
* Identify important consumption patterns and trends.
* Prepare relevant features for machine learning.
* Train a machine learning model for energy demand prediction.
* Evaluate the prediction model using appropriate performance metrics.
* Provide prediction results that can help understand future electricity demand.

---

## 🛠️ Technologies Used

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| **Python**       | Main programming language                 |
| **Apache Spark** | Big Data processing and analysis          |
| **PySpark**      | Python API for Apache Spark               |
| **Pandas**       | Data manipulation and supporting analysis |
| **Scikit-learn** | Machine learning and model evaluation     |
| **Matplotlib**   | Data visualization                        |
| **VS Code**      | Development environment                   |
| **Git & GitHub** | Version control and project management    |

---

## 📂 Dataset

### Individual Household Electric Power Consumption Dataset

The project uses the **Individual Household Electric Power Consumption** dataset.

The dataset contains measurements of electricity consumption collected over time. Some of the important attributes include:

* Date
* Time
* Global Active Power
* Global Reactive Power
* Voltage
* Global Intensity
* Sub Metering 1
* Sub Metering 2
* Sub Metering 3

The original dataset is not stored in this repository because of its size. It can be downloaded separately and placed in the project directory.

---

## 🔄 Project Workflow

The overall workflow of the project is:

```text
Raw Electricity Dataset
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Data Transformation
        ↓
Electricity Consumption Analysis
        ↓
Feature Preparation
        ↓
Model Training
        ↓
Model Testing
        ↓
Model Evaluation
        ↓
Energy Demand Prediction
```

---

## 1. Data Inspection

The first step is to understand the structure and quality of the dataset.

The data inspection process includes:

* Checking the dataset structure.
* Identifying columns and data types.
* Checking the number of records.
* Identifying missing values.
* Examining electricity consumption attributes.
* Understanding the time-based nature of the data.

---

## 2. Data Cleaning

The raw electricity dataset may contain missing or invalid values.

The data cleaning stage includes:

* Handling missing values.
* Converting required columns into appropriate data types.
* Processing date and time information.
* Removing or handling invalid records.
* Preparing the dataset for further analysis.

This step ensures that the data is suitable for analysis and machine learning.

---

## 3. Electricity Consumption Analysis

After cleaning the data, electricity consumption patterns are analyzed.

The analysis focuses on:

* Overall electricity consumption.
* Consumption trends over time.
* Changes in power usage.
* Household energy usage patterns.
* Important electricity consumption features.
* Relationships between different measurements.

Apache Spark is used to process and analyze the data efficiently.

---

## 4. Feature Engineering

The cleaned data is transformed into useful features for prediction.

Possible features include:

* Date-based features.
* Time-based features.
* Previous consumption values.
* Electricity power measurements.
* Other relevant variables derived from the original dataset.

Feature engineering helps the machine learning model identify patterns in electricity demand.

---

## 5. Model Training

The prepared dataset is divided into training and testing data.

The training data is used to train a machine learning model to learn the relationship between the input features and electricity demand.

The trained model is then used to make predictions on previously unseen test data.

---

## 6. Model Testing and Evaluation

The trained model is tested using the test dataset.

The prediction performance is evaluated using suitable metrics such as:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These metrics help measure how closely the predicted energy demand matches the actual values.

---

## 📁 Project Structure

```text
Electricity-Consumption-Spark/
│
├── analysis.py
├── app.py
├── data_cleaning.py
├── data_inspection.py
├── prediction.py
├── requirements.txt
│
├── model_metrics.csv
├── prediction_results.csv
│
├── templates/
│   └── index.html
│
├── .gitignore
│
└── README.md
```

### File Description

#### `data_inspection.py`

Used for inspecting the dataset and understanding its structure, columns, data types, and data quality.

#### `data_cleaning.py`

Handles preprocessing and cleaning of the electricity consumption data.

#### `analysis.py`

Performs electricity consumption analysis and identifies important patterns and trends.

#### `prediction.py`

Handles the machine learning workflow for training, testing, and generating energy demand predictions.

#### `app.py`

Contains the application component used to present or access the project functionality.

#### `templates/index.html`

Provides the HTML interface for the application.

#### `model_metrics.csv`

Stores the evaluation metrics generated during model evaluation.

#### `prediction_results.csv`

Stores the generated prediction results.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Hitcom-21/Electricity-Consumption-Spark.git
```

Move into the project directory:

```bash
cd Electricity-Consumption-Spark
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Download the **Individual Household Electric Power Consumption** dataset and place the extracted dataset in the project directory.

The dataset files are intentionally excluded from the GitHub repository through `.gitignore`.

---

## ▶️ Running the Project

After activating the virtual environment and installing the required dependencies, run the project scripts according to the workflow.

### Data Inspection

```bash
python data_inspection.py
```

### Data Cleaning

```bash
python data_cleaning.py
```

### Analysis

```bash
python analysis.py
```

### Prediction

```bash
python prediction.py
```

If the application component is required:

```bash
python app.py
```

---

## 📈 Output

The project generates outputs such as:

* Electricity consumption analysis results.
* Model evaluation metrics.
* Energy demand predictions.
* Prediction result files.

The generated prediction results are stored in:

```text
prediction_results.csv
```

Model evaluation results are stored in:

```text
model_metrics.csv
```

---

## 🔐 Data and Repository Management

The following files are excluded from the repository:

```text
.venv/
__pycache__/
*.pyc
individual+household+electric+power+consumption.zip
individual+household+electric+power+consumption/
```

The virtual environment and raw dataset are kept locally rather than uploaded to GitHub.

---

## 🔄 Version Control

Git is used to maintain different working versions of the project.

After making and testing changes, the project can be updated using:

```bash
git add .
git commit -m "Updated project"
git push
```

This keeps the local Git repository and GitHub repository synchronized.

---

## 🚀 Future Improvements

Possible future improvements include:

* Improving prediction accuracy.
* Testing additional machine learning algorithms.
* Adding more advanced feature engineering.
* Adding interactive visualizations.
* Improving the application interface.
* Deploying the application.
* Using larger electricity datasets for further analysis.
* Implementing real-time or near-real-time energy demand prediction.

---

## 👨‍💻 Author

**Hari Shankar**

GitHub:
https://github.com/Hitcom-21

---

## 📄 Project Type

**Academic / Big Data Analysis Project**

**Domain:** Big Data Analytics, Electricity Consumption Analysis, Machine Learning

**Primary Technology:** Apache Spark / PySpark
