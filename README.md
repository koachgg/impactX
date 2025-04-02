# Advanced Agricultural Data Analysis Pipeline

This repository contains an advanced data analysis pipeline designed to extract valuable insights from an agricultural dataset. The pipeline processes data on agricultural production and area to compute 35 advanced metrics, generate over 35 dynamic visualizations, and build a predictive model. The focus is on delivering comprehensive exploratory data analysis (EDA) and advanced visual analytics while providing a robust predictive modeling framework—even though our initial predictive modeling efforts did not yield the desired accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Functionality Details](#functionality-details)
- [Predictive Modeling Experiment](#predictive-modeling-experiment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This pipeline is tailored for analyzing agricultural data with the following columns:

- **State_Name**
- **District_Name**
- **Crop_Year**
- **Season**
- **Crop**
- **Area**
- **Production**

The pipeline performs the following tasks:
- **Data Cleaning & Preprocessing:** Cleans numeric fields and encodes categorical variables.
- **Advanced Metrics Computation:** Calculates 35 sophisticated metrics including production growth, yield trends, seasonal variability, crop diversity, and more.
- **Dynamic Visualizations:** Generates a suite of visualizations (line charts, scatter plots, box plots, sunburst, treemap, heatmap, radar, etc.) using Plotly and Matplotlib.
- **Predictive Modeling:** Implements a Random Forest model (and an experimental XGBoost approach) to predict log-scaled production values based on various features. Note that our predictive modeling experiment using XGBoost did not achieve satisfactory results, highlighting the challenges of this complex dataset.

---

## Features

- **Comprehensive Data Cleaning:**  
  Handles missing and malformed numeric values with a custom cleaning function.
  
- **Advanced Metrics:**  
  Computes 35 advanced agricultural metrics including yield trends, production growth, seasonal variability, state-wise comparisons, and forecasting.
  
- **Dynamic Visualizations:**  
  Generates 35+ interactive visualizations using Plotly, including line charts, scatter plots, violin plots, radar charts, waterfall charts, and more.
  
- **Predictive Modeling:**  
  Provides a Random Forest model pipeline for predicting production and includes an experimental XGBoost approach. Although our predictive modeling efforts with XGBoost did not yield the desired performance, we include the code for transparency and to document our iterative process.

- **Interactive Tables:**  
  Uses Plotly tables for a clear presentation of key metrics and feature importances.

---

## Requirements

The pipeline uses the following Python libraries:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [plotly](https://plotly.com/python/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/)
- Python standard libraries: `warnings`

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/agri-advanced-analysis.git
   cd agri-advanced-analysis
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *If a `requirements.txt` file is not provided, install the packages manually:*

   ```bash
   pip install pandas numpy plotly matplotlib scikit-learn xgboost
   ```

---

## Usage

1. **Prepare Your Dataset:**  
   Ensure your CSV file includes the following columns: `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`, and `Production`.

2. **Run the Pipeline:**  
   Execute the main script from the command line. You will be prompted to provide the CSV file path.

   ```bash
   python main.py
   ```

   *Example:*

   ```
   Enter CSV file path in double quotes: "data/agriculture_data.csv"
   ```

3. **Explore Results:**  
   - The script will display data information and advanced metrics.
   - Interactive visualizations will open (or be rendered inline if using a Jupyter Notebook).
   - The predictive modeling section will output model performance metrics and a table of feature importances.

---

## Code Structure

- **Data Cleaning & Preprocessing:**
  - `clean_numeric_column(series)`: Cleans numeric columns by removing unwanted characters and converting values to float.
  
- **Advanced Metrics Computation:**
  - `compute_advanced_analysis(data)`: Computes 35 advanced agricultural metrics, including yield, trends, and outlier detection.
  
- **Visualization:**
  - `create_advanced_visualizations_agri(adv_metrics, df, yearly, state_crop_analysis)`: Generates a suite of 35+ interactive visualizations using Plotly.
  - `display_dataframe_as_table(title, df)`: Displays a DataFrame as an interactive Plotly table.
  
- **Predictive Modeling:**
  - `run_predictive_model(df)`: Builds a Random Forest model to predict production (log-transformed) and displays performance metrics along with feature importances.
  
- **Pipeline Execution:**
  - `main_agri_advanced(file_path)`: Orchestrates the full pipeline: reads the dataset, computes metrics, displays visualizations, and runs the predictive model.

---

## Functionality Details

### Data Cleaning
- **Numeric Cleaning:**  
  Uses the `clean_numeric_column` function to sanitize numeric inputs and convert them to floats, addressing issues such as missing values and non-numeric entries.

### Advanced Metrics
- **Yield Calculation:**  
  Computes the yield as `Production/Area` and aggregates data yearly.
- **Trend Analysis:**  
  Calculates linear regression slopes for production and yield over the years.
- **Seasonal & State-Level Analysis:**  
  Derives statistics like seasonal variability, weighted yield averages, and crop diversity.
- **Forecasting:**  
  Uses linear regression on yearly production data to forecast production for the next year.

### Visualizations
- **Dynamic & Interactive:**  
  Over 35 visualizations are generated, including:
  - Line and area charts for trends over years.
  - Scatter, box, and violin plots for comparing variables.
  - Hierarchical visualizations like sunburst and treemap charts.
  - Advanced visualizations such as radar, waterfall, and 3D scatter plots.
- **Customization:**  
  Visualizations utilize a pleasant custom color palette for an appealing presentation.

### Predictive Modeling
- **Random Forest Regressor:**  
  Trains on log-transformed production data to handle skewness and improve model robustness.
- **Experimental XGBoost Approach:**  
  An additional predictive modeling experiment using XGBoost is provided below. While this approach did not achieve satisfactory performance, it is included for transparency and to document our iterative process.
- **Performance Metrics:**  
  Outputs RMSE, R² score, and displays feature importances in an interactive table.

---

## Predictive Modeling Experiment

Although our primary focus is on exploratory data analysis and advanced visualizations, we also attempted to build a predictive model using XGBoost. Unfortunately, the results were not satisfactory given the complexity and sparsity of the data. We include the XGBoost code below for transparency:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('apy.csv')  # Replace with your actual CSV file path

# ---- Data Cleaning ----

# 1. Strip whitespace from categorical columns to ensure consistency
categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

# 2. Replace infinite values with NaN and ensure numeric columns are in proper format
df = df.replace([np.inf, -np.inf], np.nan)
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df['Production'] = pd.to_numeric(df['Production'], errors='coerce')

# 3. Drop rows with any NaN values after conversion
df = df.dropna()

# ---- Label Encoding ----

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Classes for {col}: {le.classes_}")

# ---- Log Transformation of Target ----

# Apply a log1p transformation (log(Production + 1)) to reduce skewness
df['log_Production'] = np.log1p(df['Production'])

# ---- Splitting Data and Model Training ----

# Define features (X) and target (y: log-transformed Production)
X = df.drop(['Production', 'log_Production'], axis=1)
y = df['log_Production']

# Check target range before transformation for reference (optional)
print("Range of original Production values:")
print("Min Production:", df['Production'].min())
print("Max Production:", df['Production'].max())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set in log-space and then convert back
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Convert log predictions back to original scale
y_test_orig = np.expm1(y_test)  # Convert actual log values back to original scale

mse = mean_squared_error(y_test_orig, y_pred)
print("Mean Squared Error on test data:", mse)

# ---- Predicting on a New Sample ----

# Prepare a new sample ensuring that categorical strings are stripped of whitespace
new_sample = pd.DataFrame({
    'State_Name': ['Andaman and Nicobar Islands'],
    'District_Name': ['NICOBARS'],
    'Crop_Year': [2000],
    'Season': ['Kharif'],  # Should match training data after stripping
    'Crop': ['Arecanut'],
    'Area': [1254]
})

# Strip whitespace and transform categorical columns using the same label encoders
for col in categorical_cols:
    new_sample[col] = new_sample[col].astype(str).str.strip()
    if new_sample[col][0] not in label_encoders[col].classes_:
        raise ValueError(f"Value '{new_sample[col][0]}' for column '{col}' was not seen during training.")
    new_sample[col] = label_encoders[col].transform(new_sample[col])

# Predict log production for the new sample and convert back to original scale
predicted_log_new = model.predict(new_sample)
predicted_production_new = np.expm1(predicted_log_new)
print("Predicted Production for the new sample:", predicted_production_new[0])
```

---

## Contributing

Contributions to enhance this pipeline are welcome! Please fork the repository and submit pull requests for improvements, additional features, or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

By leveraging this advanced pipeline, you can gain deep insights into agricultural production data and build impactful predictive models to inform decisions in areas such as climate change, poverty, and public health. Although our initial predictive modeling experiment with XGBoost did not meet our expectations, it serves as a valuable learning experience and a foundation for future improvements.

Happy Analyzing!
