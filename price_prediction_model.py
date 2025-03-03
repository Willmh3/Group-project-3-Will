import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from dateutil.relativedelta import relativedelta
import os
import glob
import joblib

# ====================================================
# Part A: Data Preparation and Prophet Forecast Setup
# ====================================================

# --- Load and Prepare Housing Data (df_all) ---


# Read the main CSV file.
file_pattern = 'house_data_chunk_*.csv'

# Use glob to find all files matching the pattern
file_paths = glob.glob(file_pattern)

# Check if all 137 files are found
if len(file_paths) != 137:
    print(f"Warning: Expected 137 files, but found {len(file_paths)} files.")

# Read each file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Concatenate all DataFrames into one
df4 = pd.concat(dataframes, ignore_index=True)
df_all = df4.copy()

# --- Compute Scale Factors for Region, House_Type, and Tenure_Type ---
# These are computed on the original data before merging and dummy encoding.
df_original = df4.copy()
df_original.dropna(subset=['Date', 'Price', 'Region', 'House_Type', 'Tenure_Type'], inplace=True)
df_original['Date'] = pd.to_datetime(df_original['Date'], format='%Y-%m-%d %H:%M', errors='coerce')
df_original = df_original[df_original['Date'].notnull()]

overall_avg = df_original['Price'].mean()

region_avg = df_original.groupby('Region')['Price'].mean()
house_type_avg = df_original.groupby('House_Type')['Price'].mean()
tenure_type_avg = df_original.groupby('Tenure_Type')['Price'].mean()

region_scale_factors = region_avg / overall_avg
house_type_scale_factors = house_type_avg / overall_avg
tenure_type_scale_factors = tenure_type_avg / overall_avg

print("Computed Scale Factors:")
print("Region Scale Factors:")
print(region_scale_factors.sort_values())
print("House Type Scale Factors:")
print(house_type_scale_factors.sort_values())
print("Tenure Type Scale Factors:")
print(tenure_type_scale_factors.sort_values())

# Define the pattern to match all chunk files
file_pattern = 'UltimateRR_Chunk_*.csv'

# Use glob to find all files matching the pattern
file_paths = glob.glob(file_pattern)

# Read each file into a DataFrame and store them in a list
dataframes = [pd.read_csv(file) for file in file_paths]

# Concatenate all DataFrames into one
df_all3 = pd.concat(dataframes, ignore_index=True)
df_all = df_all.merge(df_all3[['Unique_Reference', 'tfarea', 'numberrooms', 
                               'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY']],
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
})
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
# Part C: Prediction Function
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
                               house_type=None,    # e.g., "Flat" or "Detached"
                               property_age="old", # "new" or "old"; applicable for houses
                               **encoded_kwargs):
    """
    Predicts house price using the hybrid model (Prophet base + XGBoost residual correction),
    then adjusts the forecast based on property-specific factors:
      - An adjustment based on the expected price per sqm (varies for flats vs. houses and by age)
      - A bedroom adjustment based on the number of rooms relative to a baseline of 2.95.
    Finally, the scale factors for Region, House_Type, and Tenure_Type are applied.
    
    Raises a ValueError if any inputs are invalid.
    
    Returns:
      - 'Predicted_Price': Final adjusted predicted house price.
      - 'Base_Prediction': The unadjusted prediction (for reference).
      - 'Price_Range': Tuple (lower_bound, upper_bound) which is ±5% of the final adjusted price.
      - 'Manual_Inputs': Dictionary containing the manually provided inputs.
      - 'Scale_Factors': Dictionary with scale factors for Region, House_Type, and Tenure_Type.
    """
    import pandas as pd

    # ----------------- Input Validation -----------------
    # Validate floor area and number of rooms.
    if tfarea is not None and tfarea <= 0:
        raise ValueError("Floor area (tfarea) must be a positive number.")
    if numberrooms is not None and numberrooms <= 0:
        raise ValueError("Number of rooms must be a positive integer.")

    # Validate property_age.
    allowed_property_ages = ["new", "old"]
    if property_age.lower() not in allowed_property_ages:
        raise ValueError("Invalid property_age value. Allowed values are: " + ", ".join(allowed_property_ages))
    
    # Validate house_type.
    allowed_house_types = ["flat", "detached", "semi-detached", "terraced", "house"]
    if house_type is not None:
        if house_type.lower() not in allowed_house_types:
            raise ValueError("Invalid house_type. Allowed values are: " + ", ".join(allowed_house_types))
    
    # Validate tenure_type.
    allowed_tenure_types = ["freehold", "leasehold"]
    if tenure_type is not None:
        if tenure_type.lower() not in allowed_tenure_types:
            raise ValueError("Invalid tenure_type. Allowed values are: " + ", ".join(allowed_tenure_types))
    
    # Validate region using the keys of region_scale_factors (assumed computed earlier).
    # We assume region_scale_factors has keys that are the full region names (in uppercase).
    if region is not None:
        allowed_regions = [r.upper() for r in region_scale_factors.index]
        if region.upper() not in allowed_regions:
            raise ValueError("Invalid region. Allowed regions are: " + ", ".join(allowed_regions))
    # ------------------------------------------------------

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
    input_vals['tfarea'] = tfarea if tfarea is not None else defaults['tfarea']
    input_vals['CURRENT_ENERGY_EFFICIENCY'] = CURRENT_ENERGY_EFFICIENCY if CURRENT_ENERGY_EFFICIENCY is not None else defaults['CURRENT_ENERGY_EFFICIENCY']
    input_vals['POTENTIAL_ENERGY_EFFICIENCY'] = POTENTIAL_ENERGY_EFFICIENCY if POTENTIAL_ENERGY_EFFICIENCY is not None else defaults['POTENTIAL_ENERGY_EFFICIENCY']
    input_vals['GDP_Growth_Rate'] = GDP_Growth_Rate if GDP_Growth_Rate is not None else defaults['GDP_Growth_Rate']
    input_vals['Bank_Growth_Rate'] = Bank_Growth_Rate if Bank_Growth_Rate is not None else defaults['Bank_Growth_Rate']
    input_vals['Education_Growth_Rate'] = Education_Growth_Rate if Education_Growth_Rate is not None else defaults['Education_Growth_Rate']
    input_vals['numberrooms'] = numberrooms if numberrooms is not None else defaults['numberrooms']
    
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
    
    # For encoded features: initialize with 0 for all one-hot encoded columns.
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
        desired_house = house_type.lower()
        for feat in encoded_features:
            if feat.startswith("HOUSE_TYPE_"):
                encoded_vals[feat] = 1.0 if feat.split("HOUSE_TYPE_")[1].lower() == desired_house else 0.0
    
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
    
    # Define final feature order: use all columns from monthly_data except "Date" and "Price".
    final_feature_cols = list(monthly_data.drop(columns=["Date", "Price"]).columns)
    
    # Build the input DataFrame.
    input_df = pd.DataFrame([input_vals], columns=final_feature_cols)
    
    # XGBoost residual prediction.
    xgb_res_pred = xgb_res_model.predict(input_df)[0]
    
    # Prophet base prediction.
    prophet_input = pd.DataFrame({'ds': [ds], 'Population': [Population]})
    prophet_pred = m.predict(prophet_input)['yhat'].values[0]
    
    # Base prediction is the sum of Prophet and XGBoost residual predictions.
    base_prediction = prophet_pred + xgb_res_pred

    # ----- Additional Adjustments Based on Property Specifics -----
    # 1. Price per square metre adjustment.
    # Baseline for a typical house in London is £6,900 per sqm.
    # For a flat, use £7,600; for houses, adjust by property age: "new" = £11,300, "old" = £7,100.
    if house_type is not None and house_type.lower() == "flat":
        expected_price_per_sqm = 7600
    elif house_type is not None and house_type.lower() in ["detached", "semi-detached", "terraced", "house"]:
        if property_age.lower() == "new":
            expected_price_per_sqm = 11300
        else:
            expected_price_per_sqm = 7100
    else:
        # Default to a typical house baseline if house_type is not provided.
        expected_price_per_sqm = 6900

    area_adjustment = expected_price_per_sqm / 6900.0

    # 2. Bedroom adjustment: using a baseline of 2.95 bedrooms.
    baseline_bedrooms = 2.95
    bedroom_multiplier = 0.15  # 15% increase per extra bedroom (or decrease if fewer)
    bedroom_adjustment = 1 + bedroom_multiplier * (numberrooms - baseline_bedrooms)

    # Combine the property-specific adjustments.
    property_adjustment = area_adjustment * bedroom_adjustment

    # Apply the property adjustments to the base prediction.
    intermediate_prediction = base_prediction * property_adjustment

    # ----- Apply Scale Factors (for Region, House_Type, and Tenure_Type) -----
    region_factor = region_scale_factors.get(region, 1.0) if region is not None else 1.0
    house_type_factor = house_type_scale_factors.get(house_type, 1.0) if house_type is not None else 1.0
    tenure_type_factor = tenure_type_scale_factors.get(tenure_type, 1.0) if tenure_type is not None else 1.0

    MIN_SCALE_FACTOR = 0.5  # Adjust based on domain knowledge

    # For region
    region_factor = region_scale_factors.get(region, 1.0)
    if region_factor < MIN_SCALE_FACTOR:
        region_factor = MIN_SCALE_FACTOR

    # Similarly for house_type and tenure_type
    house_type_factor = house_type_scale_factors.get(house_type, 1.0)
    if house_type_factor < MIN_SCALE_FACTOR:
        house_type_factor = MIN_SCALE_FACTOR

    tenure_type_factor = tenure_type_scale_factors.get(tenure_type, 1.0)
    if tenure_type_factor < MIN_SCALE_FACTOR:
        tenure_type_factor = MIN_SCALE_FACTOR


    scale_factors = {
        'Region': region_factor,
        'House_Type': house_type_factor,
        'Tenure_Type': tenure_type_factor
    }
    
    # Final prediction: first adjust for property specifics, then apply scale factors.
    final_adjusted_prediction = intermediate_prediction * region_factor * house_type_factor * tenure_type_factor

    # Define price range around the final adjusted prediction.
    lower_bound = final_adjusted_prediction * 0.95
    upper_bound = final_adjusted_prediction * 1.05
    
    # Build a dictionary of manually provided inputs.
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
    manual_inputs['property_age'] = property_age
    
    return {
        'Predicted_Price': final_adjusted_prediction,
        'Base_Prediction': base_prediction,
        'Price_Range': (lower_bound, upper_bound),
        'Manual_Inputs': manual_inputs,
        'Scale_Factors': scale_factors,
        'Property_Adjustments': {
            'Area_Adjustment': area_adjustment,
            'Bedroom_Adjustment': bedroom_adjustment,
            'Combined_Property_Adjustment': property_adjustment}}
    
#joblib.dump(m, 'prophet_model.pkl')
#joblib.dump(xgb_res_model, 'xgb_res_model.pkl')

#prediction = predict_house_price_hybrid(ds="2026-01-01", 
#                                        tfarea=120.0, 
#                                        numberrooms=4, 
#                                        Postcode="TW12 2DH", 
#                                        region="Wandsworth",
#                                        tenure_type="Leasehold",
#                                        house_type="Flat")
#print("\nPrediction for 2026-01-01:")
#print(prediction)
