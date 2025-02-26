import pandas as pd
import joblib

# Load the required data (assuming dataframes are preloaded)
# postcode_freq, pop_all, monthly_data, df_all

def load_models():
    """Load the trained models."""
    prophet_model = joblib.load('prophet_model.pkl')
    xgb_res_model = joblib.load('xgb_res_model.pkl')
    return prophet_model, xgb_res_model

def predict_house_price_hybrid(ds, tfarea, numberrooms, CURRENT_ENERGY_EFFICIENCY, POTENTIAL_ENERGY_EFFICIENCY,
                              Postcode, region, tenure_type, house_type, alpha=1.0):
    """
    Predicts house price using the hybrid model (Prophet base + alpha*XGBoost residual correction).
    """
    # Load models
    prophet_model, xgb_res_model = load_models()

    # Convert ds to datetime and compute time features
    ds = pd.to_datetime(ds)
    Year = ds.year
    Month = ds.month
    Year_offset = Year - 1995
    Year_offset_sq = Year_offset ** 2

    # Retrieve defaults from monthly_data (only numeric columns)
    numeric_columns = monthly_data.select_dtypes(include=['number']).columns
    monthly_data_numeric = monthly_data[numeric_columns]
    defaults = monthly_data_numeric.mean()

    input_vals = {
        'ds': ds,
        'Year': Year,
        'Month': Month,
        'Year_offset': Year_offset,
        'Year_offset_sq': Year_offset_sq,
        'tfarea': tfarea,
        'numberrooms': numberrooms,
        'CURRENT_ENERGY_EFFICIENCY': CURRENT_ENERGY_EFFICIENCY,
        'POTENTIAL_ENERGY_EFFICIENCY': POTENTIAL_ENERGY_EFFICIENCY,
        'postcode_freq': postcode_freq[postcode_freq['Postcode'] == Postcode]['Frequency'].values[0] if Postcode in postcode_freq['Postcode'].values else postcode_freq['Frequency'].mean(),
        'Population': pop_all.loc[pop_all['Year'] == Year, 'Population'].values[0]
    }

    # Add encoded features
    encoded_features = [col for col in monthly_data.columns if col.startswith('Region_') or 
                        col.startswith('Tenure_Type_') or col.startswith('House_Type_')]
    encoded_vals = {feat: 0.0 for feat in encoded_features}

    # Update for region
    if region is not None:
        desired_region = region.upper()
        for feat in encoded_features:
            if feat.startswith("Region_"):
                encoded_vals[feat] = 1.0 if feat.split("Region_")[1].upper() == desired_region else 0.0

    # Update for tenure type
    if tenure_type is not None:
        desired_tenure = tenure_type.upper()
        for feat in encoded_features:
            if feat.startswith("TENURE_TYPE_"):
                encoded_vals[feat] = 1.0 if feat.split("TENURE_TYPE_")[1].upper() == desired_tenure else 0.0

    # Update for house type
    if house_type is not None:
        desired_house = house_type.upper()
        for feat in encoded_features:
            if feat.startswith("HOUSE_TYPE_"):
                encoded_vals[feat] = 1.0 if feat.split("HOUSE_TYPE_")[1].upper() == desired_house else 0.0

    input_vals.update(encoded_vals)

    # Define final feature order
    final_feature_cols = list(monthly_data_numeric.columns)
    input_df = pd.DataFrame([input_vals], columns=final_feature_cols)

    # Predict the residual using the full feature set from XGBoost
    xgb_res_pred = xgb_res_model.predict(input_df)[0]

    # Get Prophet's base prediction
    prophet_input = pd.DataFrame({'ds': [ds], 'Population': [input_vals['Population']]})
    prophet_pred = prophet_model.predict(prophet_input)['yhat'].values[0]

    # Final weighted prediction
    final_prediction = prophet_pred + alpha * xgb_res_pred

    lower_bound = final_prediction * 0.95
    upper_bound = final_prediction * 1.05

    return {
        'Predicted_Price': float(final_prediction),
        'Price_Range': (float(lower_bound), float(upper_bound))
    }