# Solar Power Generation Data Preprocessing & ML Model

## Project Overview

This project performs comprehensive data preprocessing and machine learning modeling on solar power generation data. The workflow includes data loading, cleaning, exploratory data analysis (EDA), feature engineering, and predictive modeling using XGBoost to forecast DC power generation based on weather parameters.

## Dataset

### Input Files

1. **Plant_1_Generation_Data.csv** (68,778 rows)
   - Solar power generation data from a single plant
   - Contains multiple source keys (22 different inverters/sensors)
   - **Columns:**
     - `DATE_TIME`: Timestamp of the reading
     - `PLANT_ID`: Identifier for the power plant
     - `SOURCE_KEY`: Unique identifier for each inverter/sensor
     - `DC_POWER`: Direct current power output (Watts)
     - `AC_POWER`: Alternating current power output (Watts)
     - `DAILY_YIELD`: Cumulative energy produced in a day (Wh)
     - `TOTAL_YIELD`: Cumulative energy produced since installation (Wh)

2. **weather_data_Gandikotta.csv** (816 rows)
   - Hourly weather data for the location
   - **Columns:**
     - `temp`: Temperature (°C)
     - `humidity`: Relative humidity (%)
     - `dew`: Dew point (°C)
     - `precip`: Precipitation (mm)
     - `pressure`: Atmospheric pressure (hPa)
     - `windgust`: Wind gust speed (km/h)
     - `windspeed`: Wind speed (km/h)
     - `cloudcover`: Cloud coverage (%)
     - `timestamp`: Timestamp of the reading

## Project Workflow

### 1. Data Loading & Exploration
- Load plant generation and weather data
- Display dataset shape and information
- Identify data types and structure
- Check for missing values and duplicates

### 2. Data Cleaning
- Remove PLANT_ID column (constant value)
- Identify and handle missing values in hourly aggregated data
- Remove duplicate rows (0 found in raw data)
- Handle missing values in resampled data using time-based imputation

### 3. Outlier Detection (IQR Method)
- Calculate Q1, Q3, and IQR for numerical columns
- Identify outliers beyond ±1.5×IQR bounds
- Result: No outliers detected in the dataset

### 4. Data Segmentation by Source
- Split data by SOURCE_KEY (22 inverters)
- Analyze row counts per source (ranging from 3,104 to 3,155 rows)
- Select source with maximum rows (3,155 rows) for further analysis
- All sources have no missing values

### 5. Feature Engineering
- **Time-based resampling:** Aggregate raw data to hourly frequency
  - From 68,778 15-minute readings to 816 hourly readings
  - Calculate mean values for numerical features
- **Missing value imputation:** 
  - Use mean for times closer to noon
  - Use minimum for times closer to midnight
  - Result: All 20 missing values imputed

### 6. Data Merging
- Merge hourly aggregated generation data with weather data
- Use inner join on DATE_TIME and timestamp columns
- Final merged dataset: 816 rows × 14 columns
- All merged data has no missing values

### 7. Normality Assessment
- **Shapiro-Wilk Test Results:**
  - DC_POWER: p-value = 0.0000 (not normally distributed)
  - AC_POWER: p-value = 0.0000 (not normally distributed)
  - DAILY_YIELD: p-value = 0.0000 (not normally distributed)
  - TOTAL_YIELD: p-value = 0.0000 (not normally distributed)
- **Skewness values:** Positive for power metrics (0.96), negative for TOTAL_YIELD (-0.42)

### 8. Feature Scaling
Applied two scaling approaches:
- **StandardScaler:** Z-score normalization (mean=0, std=1)
- **RobustScaler:** Robust to outliers using IQR-based scaling

### 9. Correlation Analysis
Analyzed correlation between weather features and DC_POWER:
- **Temperature:** 0.30 (positive correlation)
- **Humidity:** -0.20 (negative correlation)
- **Cloudcover:** -0.09 (weak negative correlation)
- **Precipitation:** -0.10 (weak negative correlation)

**Key Finding:** Temperature is the most influential weather feature (|r| > 0.2)

### 10. Data Splitting
- **Training set:** 652 samples (80%)
- **Testing set:** 164 samples (20%)
- Random state: 42 (reproducible splits)

### 11. Machine Learning Model

#### Model: XGBoost Regressor
**Hyperparameters:**
- Objective: reg:squarederror
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 5
- random_state: 42

**Features (X):**
- Temperature (temp)
- Humidity
- Cloud cover (cloudcover)
- Precipitation (precip)

**Target (y):**
- DC_POWER (Direct Current Power)

### 12. Model Performance

#### Training Metrics
| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | 1,396.68 W |
| MSE (Mean Squared Error) | 3,389,751.08 |
| RMSE (Root Mean Squared Error) | 1,841.13 W |
| R² (R-squared) | 0.7378 |

#### Testing Metrics
| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | 2,723.47 W |
| MSE (Mean Squared Error) | 11,369,238.51 |
| RMSE (Root Mean Squared Error) | 3,372.43 W |
| R² (R-squared) | 0.0903 |

#### Model Generalization Analysis
- **MAE Difference (Test - Train):** +1,326.79 W
- **RMSE Difference (Test - Train):** +1,531.30 W
- **R² Difference (Train - Test):** +0.6475
- **Assessment:** ⚠️ Significant overfitting detected

### 13. Residual Error Analysis

#### Training Residuals Statistics
- **Mean:** 2.64 W (near zero, indicating unbiased predictions)
- **Standard Deviation:** 1,842.54 W
- **Min:** -5,335.84 W
- **Max:** 6,497.44 W
- **Median:** -296.09 W

#### Residual Distribution
- **Shapiro-Wilk Test:** p-value = 1.25e-11
- **Result:** Residuals NOT normally distributed
- **Q-Q Plot:** Shows deviation from normality at tails

#### Outlier Detection
- **3σ Threshold:** ±5,527.62 W
- **Outliers Found:** 4 (0.61% of training data)
- **Top Outliers:**
  1. Position 67: Residual = 6,497.44 W (Actual: 11,056.26 W, Predicted: 4,558.81 W)
  2. Position 192: Residual = 5,865.97 W (Actual: 10,442.36 W, Predicted: 4,576.39 W)
  3. Position 508: Residual = 5,840.84 W (Actual: 10,949.43 W, Predicted: 5,108.59 W)
  4. Position 46: Residual = 5,799.08 W (Actual: 11,134.92 W, Predicted: 5,335.84 W)

#### Residual Percentiles
- 1st: -4,009.41 W
- 5th: -2,575.75 W
- 25th: -1,198.61 W
- 50th (Median): -296.09 W
- 75th: 914.20 W
- 95th: 3,540.70 W
- 99th: 5,171.03 W

#### Residuals vs Fitted Values Analysis
- Pattern observed: Higher variability at higher fitted values
- Suggests heteroscedasticity (non-constant variance)
- Model assumptions may be violated for extreme values

## Key Findings

### Data Insights
1. **Multi-source System:** Data from 22 different inverters with balanced row counts
2. **No Data Quality Issues:** No missing values, duplicates, or outliers in raw data
3. **Seasonal Variation:** TOTAL_YIELD shows monotonic increase (negative skew)
4. **Weather Dependency:** Temperature is the primary weather feature affecting power output

### Model Insights
1. **Moderate Training Performance:** R² of 0.74 indicates decent fit but considerable unexplained variance
2. **Poor Generalization:** R² drops to 0.09 on test set, indicating severe overfitting
3. **Weather Features Limitation:** Using only weather features captures limited relationship with power
4. **Model Limitations:** 
   - Does not account for time-of-day effects
   - Missing crucial features (e.g., solar irradiance, panel temperature, angle)
   - Non-linear relationships not fully captured

## Recommendations for Improvement

### Data Collection
- Add solar irradiance measurements (most critical for solar power prediction)
- Include panel surface temperature
- Collect panel tilt angle and orientation information
- Record maintenance and cleaning events

### Feature Engineering
- Add time-based features (hour, day of year, season)
- Create interaction features between weather variables
- Include lagged power values (autoregressive components)
- Calculate solar position (elevation, azimuth)

### Model Improvements
- Try ensemble methods (Random Forest, Gradient Boosting with tuned parameters)
- Implement deep learning (LSTM for temporal patterns)
- Use separate models for different operational modes (night/day)
- Apply hyperparameter optimization (GridSearch, RandomSearch)

### Address Overfitting
- Use regularization (L1/L2 penalty)
- Implement cross-validation
- Increase training data or reduce model complexity
- Consider early stopping during training

## Project Structure

```
workspace/
├── README.md                              # This file
├── Data_Preprocessing (2).ipynb          # Main analysis notebook
├── Plant_1_Generation_Data.csv           # Raw generation data
└── weather_data_Gandikotta.csv           # Weather data
```

## Dependencies

```python
numpy                 # Numerical computing
pandas                # Data manipulation and analysis
matplotlib            # Data visualization
seaborn               # Statistical data visualization
scikit-learn          # Machine learning library
  - StandardScaler
  - RobustScaler
  - train_test_split
  - mean_absolute_error
  - mean_squared_error
  - r2_score
scipy                 # Statistical analysis
xgboost               # Gradient boosting framework
```

## Usage

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost
   ```

2. **Run the Analysis:**
   - Open `Data_Preprocessing (2).ipynb` in Jupyter Notebook or JupyterLab
   - Ensure data files are in the same directory
   - Execute cells sequentially from top to bottom
   - Or run all cells at once using Kernel > Restart & Run All

3. **View Results:**
   - Notebook displays statistical summaries and visualizations
   - Check console outputs for model metrics
   - Review residual analysis plots for model diagnostics

## Conclusions

This project successfully demonstrates a complete data science pipeline from raw data to predictive modeling. While the XGBoost model achieves reasonable training performance (R² = 0.74), the significant drop in test performance indicates that weather features alone are insufficient for accurate solar power prediction. The analysis highlights the importance of domain-specific features (solar irradiance) and temporal patterns in solar energy forecasting.

The residual analysis reveals that model predictions tend to underestimate high power generation events, suggesting the need for additional features that capture peak generation potential, such as direct solar irradiance measurements.

---

**Last Updated:** October 20, 2025  
**Analysis Tool:** Jupyter Notebook with Python  
**Data Location:** c:\Users\GIGABYTE\Downloads\workspace\
