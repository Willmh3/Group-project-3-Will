import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from dateutil.relativedelta import relativedelta
import os
import glob

# ====================================================
# Part A: Data Preparation and Prophet Forecast Setup
# ====================================================

# --- Load and Prepare Housing Data (df_all) ---

# Define the pattern to match all chunk files
file_pattern = 'UltimateRR_Chunk_*.csv'

# Use glob to find all files matching the pattern
file_paths = glob.glob(file_pattern)

# Read each file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Concatenate all DataFrames into one
df4 = pd.concat(dataframes, ignore_index=True)

# Define the pattern to match all chunk files
file_pattern = 'house_data_chunk_*.csv'

# Use glob to find all files matching the pattern
file_paths = glob.glob(file_pattern)

# Check if all 137 files are found
if len(file_paths) != 137:
    print(f"Warning: Expected 137 files, but found {len(file_paths)} files.")

# Read each file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
df_all = combined_df.copy()


df_all = df_all.merge(df4[['Unique_Reference', 'tfarea', 'numberrooms', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY']],
                     on='Unique_Reference', how='left')
df_all.drop(columns=["ID", "Unique_Reference", "House_Number", "Flat_Number", "A", "A.1"], inplace=True)
df_all.dropna(subset=['Date', 'Price', 'tfarea', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY'], inplace=True)

df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d %H:%M', errors='coerce')
df_all.dropna(subset=['Date'], inplace=True)
df_all.sort_values(by='Date', inplace=True)
df_all.reset_index(drop=True, inplace=True)

# --- Incorporate Non-Numeric Data ---
df_all['postcode_freq'] = df_all['Postcode'].map(df_all['Postcode'].value_counts(normalize=True))
df_all = pd.get_dummies(df_all, columns=['Region', 'Tenure_Type', 'House_Type'], drop_first=True)

# Remove outliers by year.
df_all['Year'] = df_all['Date'].dt.year
def remove_outliers(group):
    Q1 = group['Price'].quantile(0.25)
    Q3 = group['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['Price'] >= lower_bound) & (group['Price'] <= upper_bound)]
df_all = df_all.groupby('Year', group_keys=False).apply(remove_outliers)

# ===============================
# Monthly Aggregation & Time Features
# ===============================
df_all.set_index('Date', inplace=True)
monthly_data = df_all.resample('M').mean(numeric_only=True).reset_index()
monthly_data['Year'] = monthly_data['Date'].dt.year
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['Year_offset'] = monthly_data['Year'] - 1995
monthly_data['Year_offset_sq'] = monthly_data['Year_offset'] ** 2

# ===============================
# Incorporate London Population Data
# ===============================
pop_data = pd.DataFrame({
    'Year': [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016,
             2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007,
             2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998,
             1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990],
    'Population': [9841000, 9748000, 9648000, 9541000, 9426000, 9304000,
                   9177000, 9046000, 8916000, 8788000, 8661000, 8537000,
                   8414000, 8293000, 8174000, 8044000, 7917000, 7792000,
                   7668000, 7547000, 7501000, 7456000, 7411000, 7367000,
                   7322000, 7273000, 7223000, 7174000, 7126000, 7078000,
                   7030000, 6982000, 6935000, 6888000, 6841000, 6794000]
}) #################################################################################### https://www.macrotrends.net/global-metrics/cities/22860/london/population- 25/02/2025 12:26
pop_data = pop_data.sort_values('Year').reset_index(drop=True)
growth_rate_2025 = 0.0095
max_year = pop_data['Year'].max()
future_years = list(range(max_year + 1, 2041))
future_population = []
last_pop = pop_data.loc[pop_data['Year'] == max_year, 'Population'].values[0]
for yr in future_years:
    last_pop = last_pop * (1 + growth_rate_2025)
    future_population.append(last_pop)
pop_future = pd.DataFrame({'Year': future_years, 'Population': future_population})
pop_all = pd.concat([pop_data, pop_future], ignore_index=True)
monthly_data = pd.merge(monthly_data, pop_all, on='Year', how='left')

# ===============================
# Prepare Data for Prophet
# ===============================
prophet_df = monthly_data[['Date', 'Price', 'Population']].copy()
prophet_df.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)

# ===============================
# Fit Prophet Model (with Population as a Regressor)
# ===============================
m = Prophet()
m.add_regressor('Population')
m.fit(prophet_df)

# ===============================
# Create Future DataFrame for Prophet Until 2040
# ===============================
max_year_hist = prophet_df['ds'].dt.year.max()
months_to_add = (2040 - max_year_hist + 1) * 12
future_extended = m.make_future_dataframe(periods=months_to_add, freq='MS')
future_extended['Year'] = future_extended['ds'].dt.year
future_extended = future_extended.merge(pop_all, on='Year', how='left')
future_extended['Population'] = future_extended['Population'].fillna(method='ffill')
forecast_prophet = m.predict(future_extended)
prophet_future = forecast_prophet[forecast_prophet['ds'] > prophet_df['ds'].max()][['ds', 'yhat']]
prophet_future.rename(columns={'ds': 'Date', 'yhat': 'Prophet_Forecast'}, inplace=True)

# ====================================================
# Part B: Train XGBoost Residual Model on ALL Parameters
# ====================================================
# Compute Prophet in-sample predictions on historical data.
prophet_in_sample = m.predict(prophet_df[['ds', 'Population']])
# Compute residuals: actual Price - Prophet prediction.
residuals = prophet_df['y'] - prophet_in_sample['yhat']

# Build residual training dataset using all numeric predictors from monthly_data except Date and Price.
X_res = monthly_data.drop(columns=["Date", "Price"])
y_res = residuals.values  # Residuals corresponding to each monthly row.

# Train the XGBoost residual model on all available features.
xgb_res_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    random_state=42
)
xgb_res_model.fit(X_res, y_res)

# Save the list of features used in the residual model.
residual_feature_cols = list(X_res.columns)

# ====================================================
# Part C: Weighted Prediction Function
# ====================================================
def predict_house_price_hybrid(ds="2025-01-01", 
                               tfarea=None, 
                               CURRENT_ENERGY_EFFICIENCY=None, 
                               POTENTIAL_ENERGY_EFFICIENCY=None,
                               GDP_Growth_Rate=None, 
                               Bank_Growth_Rate=None, 
                               Education_Growth_Rate=None,
                               numberrooms=None, 
                               Postcode=None,      # e.g., "TW12 2DH"
                               postcode_freq=None, 
                               Population=None,
                               region=None,        # e.g., "Wandsworth"
                               tenure_type=None,   # e.g., "Leasehold"
                               house_type=None,    # e.g., "Flat"
                               alpha=1.0,          # Weight for residual correction.
                               **encoded_kwargs):
    """
    Predicts house price using the hybrid model (Prophet base + alpha*XGBoost residual correction)
    and returns:
      - 'Predicted_Price': Final predicted house price.
      - 'Price_Range': Tuple (lower_bound, upper_bound) which is ±5% of predicted price.
      - 'Manual_Inputs': Dictionary containing only manually provided inputs (with non-numeric features 
                         in their original form, e.g., Postcode, region, tenure type, house type).
    
    NOTE:
      The Prophet model uses only ds and Population.
      The XGBoost residual model is trained on ALL numeric predictors from monthly_data (i.e., all columns 
      except Date and Price). Therefore, if you change any manual inputs that are part of that full set, 
      the final prediction will change accordingly.
      
    Parameters:
      ds (str or pd.Timestamp): Prediction date (e.g., '2025-01-01').
      tfarea, CURRENT_ENERGY_EFFICIENCY, POTENTIAL_ENERGY_EFFICIENCY,
      GDP_Growth_Rate, Bank_Growth_Rate, Education_Growth_Rate, numberrooms:
          Numeric features (defaults from monthly_data if not provided).
      Postcode (str): Original postcode (e.g., "TW12 2DH"). If provided, used to compute postcode_freq.
      postcode_freq (float): Frequency-encoded postcode value.
      Population (float): London population for the given year.
      region (str): Region as a string (e.g., "Wandsworth").
      tenure_type (str): Tenure type as a string (e.g., "Leasehold").
      house_type (str): House type as a string (e.g., "Flat").
      alpha (float): Weight (0 to 1) for the residual correction.
      **encoded_kwargs: Additional one-hot encoded feature values.
      
    Returns:
      dict: Contains:
           - 'Predicted_Price': Final predicted house price.
           - 'Price_Range': (lower_bound, upper_bound) ±5% of predicted price.
           - 'Manual_Inputs': Dictionary of only manually provided inputs (Population is omitted).
    """
    import pandas as pd
    
    # Convert ds to datetime and compute time features.
    ds = pd.to_datetime(ds)
    Year = ds.year
    Month = ds.month
    Year_offset = Year - 1995
    Year_offset_sq = Year_offset ** 2
    
    # Retrieve defaults from monthly_data.
    defaults = monthly_data.mean()
    input_vals = {}
    
    # Record time features.
    input_vals['ds'] = ds
    input_vals['Year'] = Year
    input_vals['Month'] = Month
    input_vals['Year_offset'] = Year_offset
    input_vals['Year_offset_sq'] = Year_offset_sq
    
    # Process numeric features.
    if tfarea is None:
        tfarea = defaults['tfarea']
    input_vals['tfarea'] = tfarea
    
    if CURRENT_ENERGY_EFFICIENCY is None:
        CURRENT_ENERGY_EFFICIENCY = defaults['CURRENT_ENERGY_EFFICIENCY']
    input_vals['CURRENT_ENERGY_EFFICIENCY'] = CURRENT_ENERGY_EFFICIENCY
    
    if POTENTIAL_ENERGY_EFFICIENCY is None:
        POTENTIAL_ENERGY_EFFICIENCY = defaults['POTENTIAL_ENERGY_EFFICIENCY']
    input_vals['POTENTIAL_ENERGY_EFFICIENCY'] = POTENTIAL_ENERGY_EFFICIENCY
    
    if GDP_Growth_Rate is None:
        GDP_Growth_Rate = defaults['GDP_Growth_Rate']
    input_vals['GDP_Growth_Rate'] = GDP_Growth_Rate
    
    if Bank_Growth_Rate is None:
        Bank_Growth_Rate = defaults['Bank_Growth_Rate']
    input_vals['Bank_Growth_Rate'] = Bank_Growth_Rate
    
    if Education_Growth_Rate is None:
        Education_Growth_Rate = defaults['Education_Growth_Rate']
    input_vals['Education_Growth_Rate'] = Education_Growth_Rate
    
    if numberrooms is None:
        numberrooms = defaults['numberrooms']
    input_vals['numberrooms'] = numberrooms
    
    # Process postcode frequency.
    if postcode_freq is None:
        if Postcode is not None:
            freq_series = df_all['Postcode'].value_counts(normalize=True)
            postcode_freq = freq_series.get(Postcode, monthly_data['postcode_freq'].mean() if 'postcode_freq' in monthly_data.columns else 0.0)
        else:
            postcode_freq = monthly_data['postcode_freq'].mean() if 'postcode_freq' in monthly_data.columns else 0.0
    input_vals['postcode_freq'] = postcode_freq
    
    # Process Population.
    if Population is None:
        pop_val = pop_all.loc[pop_all['Year'] == Year, 'Population']
        if not pop_val.empty:
            Population = pop_val.values[0]
        else:
            Population = monthly_data['Population'].iloc[-1]
    input_vals['Population'] = Population
    
    # For encoded features: initialize with 0.
    encoded_features = [col for col in monthly_data.columns if col.startswith('Region_') or 
                        col.startswith('Tenure_Type_') or col.startswith('House_Type_')]
    encoded_vals = {feat: 0.0 for feat in encoded_features}
    for feat in encoded_features:
        if feat in encoded_kwargs:
            encoded_vals[feat] = encoded_kwargs[feat]
    
    # Update for region.
    if region is not None:
        desired_region = region.upper()
        for feat in encoded_features:
            if feat.startswith("Region_"):
                encoded_vals[feat] = 1.0 if feat.split("Region_")[1].upper() == desired_region else 0.0
    # Update for tenure type.
    if tenure_type is not None:
        desired_tenure = tenure_type.upper()
        for feat in encoded_features:
            if feat.startswith("TENURE_TYPE_"):
                encoded_vals[feat] = 1.0 if feat.split("TENURE_TYPE_")[1].upper() == desired_tenure else 0.0
    # Update for house type.
    if house_type is not None:
        desired_house = house_type.upper()
        for feat in encoded_features:
            if feat.startswith("HOUSE_TYPE_"):
                encoded_vals[feat] = 1.0 if feat.split("HOUSE_TYPE_")[1].upper() == desired_house else 0.0
    
    input_vals.update(encoded_vals)
    
    # Record original non-numeric inputs.
    if Postcode is not None:
        input_vals['Postcode'] = Postcode
    if region is not None:
        input_vals['Region'] = region
    if tenure_type is not None:
        input_vals['Tenure_Type'] = tenure_type
    if house_type is not None:
        input_vals['House_Type'] = house_type
    
    # Define final feature order: use all numeric columns from monthly_data (i.e., all columns except "Date" and "Price").
    final_feature_cols = list(monthly_data.drop(columns=["Date", "Price"]).columns)
    input_df = pd.DataFrame([input_vals], columns=final_feature_cols)
    
    # Predict the residual using the full feature set from XGBoost.
    xgb_res_pred = xgb_res_model.predict(input_df)[0]
    
    # Get Prophet's base prediction.
    prophet_input = pd.DataFrame({'ds': [ds], 'Population': [Population]})
    prophet_pred = m.predict(prophet_input)['yhat'].values[0]
    
    # Final weighted prediction.
    final_prediction = prophet_pred + alpha * xgb_res_pred
    
    lower_bound = final_prediction * 0.95
    upper_bound = final_prediction * 1.05
    
    # Build a dictionary of manually provided inputs (excluding Population).
    manual_inputs = {}
    if ds != pd.to_datetime("2025-01-01"):
        manual_inputs['ds'] = ds.strftime("%Y-%m-%d")
    if tfarea != defaults['tfarea']:
        manual_inputs['tfarea'] = tfarea
    if CURRENT_ENERGY_EFFICIENCY != defaults['CURRENT_ENERGY_EFFICIENCY']:
        manual_inputs['CURRENT_ENERGY_EFFICIENCY'] = CURRENT_ENERGY_EFFICIENCY
    if POTENTIAL_ENERGY_EFFICIENCY != defaults['POTENTIAL_ENERGY_EFFICIENCY']:
        manual_inputs['POTENTIAL_ENERGY_EFFICIENCY'] = POTENTIAL_ENERGY_EFFICIENCY
    if GDP_Growth_Rate != defaults['GDP_Growth_Rate']:
        manual_inputs['GDP_Growth_Rate'] = GDP_Growth_Rate
    if Bank_Growth_Rate != defaults['Bank_Growth_Rate']:
        manual_inputs['Bank_Growth_Rate'] = Bank_Growth_Rate
    if Education_Growth_Rate != defaults['Education_Growth_Rate']:
        manual_inputs['Education_Growth_Rate'] = Education_Growth_Rate
    if numberrooms != defaults['numberrooms']:
        manual_inputs['numberrooms'] = numberrooms
    if postcode_freq != (monthly_data['postcode_freq'].mean() if 'postcode_freq' in monthly_data.columns else 0.0):
        manual_inputs['postcode_freq'] = postcode_freq
    if Postcode is not None:
        manual_inputs['Postcode'] = Postcode
    if region is not None:
        manual_inputs['Region'] = region
    if tenure_type is not None:
        manual_inputs['Tenure_Type'] = tenure_type
    if house_type is not None:
        manual_inputs['House_Type'] = house_type
    
    return {
        'Predicted_Price': final_prediction,
        'Price_Range': (lower_bound, upper_bound),
        'Manual_Inputs': manual_inputs
    }

# Example usage:
prediction = predict_house_price_hybrid(ds="2026-01-01", 
                                        tfarea=120.0, 
                                        numberrooms=4, 
                                        Postcode="TW12 2DH", 
                                        region="Wandsworth",
                                        tenure_type="Leasehold",
                                        house_type="Flat",
                                        alpha=0.7)
print("\nPrediction for 2026-01-01:")
print(prediction)
