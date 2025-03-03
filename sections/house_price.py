import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import glob
import os

# Load the saved models
@st.cache_resource
def load_models():
    prophet_model = joblib.load('prophet_model.pkl')
    xgb_res_model = joblib.load('xgb_res_model.pkl')
    return prophet_model, xgb_res_model

# Load only the necessary columns from FinalData.parquet
@st.cache_data
def load_data():
    # Only read specific columns instead of the entire file
    columns_to_read = [
        'postcode', 'Street_Name', 'House_Number', 'Flat_Number', 
        'numberOfBedrooms', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
        'tfarea', 'latitude', 'longitude', '£PSqFt', 'borough', 'house_Type'
    ]
    
    # Read the parquet file in chunks to reduce memory usage
    return pd.read_parquet('FinalData.parquet', columns=columns_to_read)

# Load population data
@st.cache_data
def load_population_data():
    # Define the population data
    pop_all = pd.DataFrame({
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

    # Extend population data to future years (up to 2040)
    growth_rate_2025 = 0.0095
    max_year = pop_all['Year'].max()
    future_years = list(range(max_year + 1, 2041))
    future_population = []
    last_pop = pop_all.loc[pop_all['Year'] == max_year, 'Population'].values[0]
    
    for yr in future_years:
        last_pop = last_pop * (1 + growth_rate_2025)
        future_population.append(last_pop)

    # Combine historical and future population data
    pop_future = pd.DataFrame({'Year': future_years, 'Population': future_population})
    return pd.concat([pop_all, pop_future], ignore_index=True)

# Load postcode frequency data
@st.cache_data
def load_postcode_freq():
    return pd.read_csv('postcode_freq.csv')

@st.cache_data
def load_scale_factors():
    return joblib.load('scale_factors.pkl')

# Load the encoded features for model prediction
@st.cache_data
def load_encoded_features():
    try:
        # Read a small sample to get the column names
        monthly_data = pd.read_csv('monthly_data.csv')
        return monthly_data
    except FileNotFoundError:
        # If the file doesn't exist, return a default list of encoded features
        st.warning("monthly_data.csv not found. Using default encoded features.")
        return ['Region_LONDON', 'House_Type_DETACHED', 'House_Type_FLAT', 'House_Type_SEMI', 'House_Type_TERRACED']

# Extract features from the dataset
def extract_features(df, postcode, street_name=None, house_number=None, flat_number=None):
    """
    Extract features from the dataset based on user inputs.
    If only postcode is provided, compute average values for that postcode.
    """
    # Filter by postcode first (most selective filter)
    filtered_df = df[df['postcode'] == postcode]
    
    # Apply additional filters if provided
    if street_name:
        filtered_df = filtered_df[filtered_df['Street_Name'] == street_name]
    if house_number:
        filtered_df = filtered_df[filtered_df['House_Number'] == house_number]
    if flat_number:
        filtered_df = filtered_df[filtered_df['Flat_Number'] == flat_number]

    # Return None if no matching records
    if filtered_df.empty:
        return None

    # Compute average values for the numeric columns
    numeric_cols = ['numberOfBedrooms', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
                   'tfarea', 'latitude', 'longitude', '£PSqFt']
    avg_numeric = filtered_df[numeric_cols].mean()
    
    # For categorical columns, take the most common value (mode)
    categorical_cols = ['borough', 'house_Type']
    mode_categorical = {}
    
    for col in categorical_cols:
        if col in filtered_df.columns and not filtered_df[col].empty:
            mode_val = filtered_df[col].mode().iloc[0]
            mode_categorical[col] = mode_val
        else:
            mode_categorical[col] = None
    
    # Combine numeric and categorical values
    result = {**avg_numeric.to_dict(), **mode_categorical}
    
    return pd.Series(result)

# Define the prediction function
def predict_house_price_hybrid(ds, numberrooms, Postcode, region, house_type, tfarea, 
                               CURRENT_ENERGY_EFFICIENCY, POTENTIAL_ENERGY_EFFICIENCY, 
                               postcode_freq_data, pop_all_data, encoded_features, 
                               prophet_model, xgb_res_model, property_age="old", tenure_type="freehold",
                               scale_factors=None):
    """
    Predicts house price using the hybrid model (Prophet base + XGBoost residual correction),
    then adjusts the forecast based on property-specific factors.
    """
    # Convert ds to datetime and compute time features
    ds = pd.to_datetime(ds)
    Year = ds.year
    Month = ds.month
    Year_offset = Year - 1995
    Year_offset_sq = Year_offset ** 2

    # Get postcode frequency
    if Postcode in postcode_freq_data['Postcode'].values:
        freq = postcode_freq_data[postcode_freq_data['Postcode'] == Postcode]['Frequency'].values[0]
    else:
        freq = postcode_freq_data['Frequency'].mean()
    
    # Get population for the year
    if Year in pop_all_data['Year'].values:
        population = pop_all_data.loc[pop_all_data['Year'] == Year, 'Population'].values[0]
    else:
        # Use latest population with growth rate if year not found
        latest_year = pop_all_data['Year'].max()
        latest_pop = pop_all_data.loc[pop_all_data['Year'] == latest_year, 'Population'].values[0]
        growth_rate_2025 = 0.0095
        population = latest_pop * (1 + growth_rate_2025) ** (Year - latest_year)
    
    # Add the missing economic indicators (using default values based on recent averages)
    education_growth_rate = 0.025
    bank_growth_rate = 0.015
    gdp_growth_rate = 0.02

    # Create a dictionary with all features needed for prediction
    features_dict = {
        'Education_Growth_Rate': education_growth_rate,
        'Bank_Growth_Rate': bank_growth_rate,
        'GDP_Growth_Rate': gdp_growth_rate,
        'tfarea': tfarea,
        'numberrooms': numberrooms,
        'CURRENT_ENERGY_EFFICIENCY': CURRENT_ENERGY_EFFICIENCY,
        'POTENTIAL_ENERGY_EFFICIENCY': POTENTIAL_ENERGY_EFFICIENCY,
        'postcode_freq': freq,
        'Year': Year,
        'Month': Month,
        'Year_offset': Year_offset,
        'Year_offset_sq': Year_offset_sq,
        'Population': population
    }
    
    # Initialize all region features to 0
    for feat in encoded_features:
        if feat.startswith("Region_"):
            features_dict[feat] = 0.0
    
    # Set the correct region to 1 if available
    if region is not None:
        desired_region = region.upper()
        region_feature = f"Region_{desired_region}"
        if region_feature in encoded_features:
            features_dict[region_feature] = 1.0
    
    # Initialize all house type features to 0
    for feat in encoded_features:
        if feat.startswith("House_Type_") or feat.startswith("Tenure_Type_"):
            features_dict[feat] = 0.0
    
    # Set the correct house type to 1 if available
    if house_type is not None:
        desired_house = house_type.upper()
        # Handle different formats (full name vs first letter)
        if f"House_Type_{desired_house}" in encoded_features:
            features_dict[f"House_Type_{desired_house}"] = 1.0
        elif f"House_Type_{desired_house[0]}" in encoded_features:  # First letter only (F, S, T, etc.)
            features_dict[f"House_Type_{desired_house[0]}"] = 1.0
    
    # Create input DataFrame for XGBoost
    xgb_input_df = pd.DataFrame([features_dict])
    
    # Get the list of features expected by the model
    try:
        # For newer XGBoost versions
        model_features = xgb_res_model.feature_names
    except AttributeError:
        try:
            # For older XGBoost versions
            model_features = xgb_res_model.feature_names_in_
        except AttributeError:
            # If both approaches fail, we'll need to extract from the error message
            st.warning("Could not determine model features directly. Using a simplified approach.")
            model_features = list(features_dict.keys())
    
    # Ensure all model features are present in the input
    for feature in model_features:
        if feature not in xgb_input_df.columns:
            xgb_input_df[feature] = 0.0  # Add missing features with default values
    
    # Select only the features the model expects and in the right order
    xgb_input_df = xgb_input_df[model_features]
    
    # Predict the residual using XGBoost (convert to numpy array to ensure compatibility)
    xgb_res_pred = xgb_res_model.predict(xgb_input_df.values)[0]

    # Create separate input DataFrame for Prophet (only needs ds and Population)
    prophet_input = pd.DataFrame({'ds': [ds], 'Population': [population]})
    prophet_pred = prophet_model.predict(prophet_input)['yhat'].values[0]

    # Base prediction is the sum of Prophet and XGBoost residual predictions
    base_prediction = prophet_pred + xgb_res_pred

    # ----- Additional Adjustments Based on Property Specifics -----
    # 1. Price per square metre adjustment
    if house_type is not None and house_type.lower() == "flat":
        expected_price_per_sqm = 7600
    elif house_type is not None and house_type.lower() in ["detached", "semi-detached", "terraced", "house"]:
        if property_age.lower() == "new":
            expected_price_per_sqm = 11300
        else:
            expected_price_per_sqm = 7100
    else:
        # Default to a typical house baseline if house_type is not provided
        expected_price_per_sqm = 6900

    area_adjustment = expected_price_per_sqm / 6900.0

    # 2. Bedroom adjustment: using a baseline of 2.95 bedrooms
    baseline_bedrooms = 2.95
    bedroom_multiplier = 0.15  # 15% increase per extra bedroom (or decrease if fewer)
    bedroom_adjustment = 1 + bedroom_multiplier * (numberrooms - baseline_bedrooms)

    # Combine the property-specific adjustments
    property_adjustment = area_adjustment * bedroom_adjustment

    # Apply the property adjustments to the base prediction
    intermediate_prediction = base_prediction * property_adjustment

    # ----- Apply Scale Factors (for Region, House_Type, and Tenure_Type) -----
    if scale_factors:
        region_factor = scale_factors['region_scale_factors'].get(region, 1.0)
        house_type_factor = scale_factors['house_type_scale_factors'].get(house_type, 1.0)
        tenure_type_factor = scale_factors['tenure_type_scale_factors'].get(tenure_type, 1.0)
    else:
        region_factor = 1.0
        house_type_factor = 1.0
        tenure_type_factor = 1.0

    # Final prediction: first adjust for property specifics, then apply scale factors
    final_adjusted_prediction = intermediate_prediction * region_factor * house_type_factor * tenure_type_factor

    # Define price range around the final adjusted prediction
    lower_bound = final_adjusted_prediction * 0.95
    upper_bound = final_adjusted_prediction * 1.05
    
    return {
        'Predicted_Price': final_adjusted_prediction,
        'Price_Range': (lower_bound, upper_bound)
    }
    
def show():
    st.title("House Price Prediction")
    
    try:
        # Load data and models at app startup
        with st.spinner("Loading models and data..."):
            prophet_model, xgb_res_model = load_models()
            df = load_data()
            pop_all_data = load_population_data()
            postcode_freq_data = load_postcode_freq()
            encoded_features = load_encoded_features()
            scale_factors = load_scale_factors()  # Load scale factors
        
        # Input form
        with st.form("house_price_form"):
            st.write("Enter the details of the house:")
            
            # Collect user inputs
            postcode = st.text_input("Postcode").upper()
            street_name = st.text_input("Street Name (optional)")
            house_number = st.text_input("House Number (optional)")
            flat_number = st.text_input("Flat Number (optional)")
            
            submitted = st.form_submit_button("Predict Price")

        if submitted:
            if not postcode:
                st.error("Postcode is required.")
                return

            with st.spinner("Extracting features..."):
                # Extract features based on user inputs
                features = extract_features(df, postcode, street_name, house_number, flat_number)

            if features is not None:
                st.subheader("Property Details")
                feature_display = {
                    "Number of Bedrooms": round(features['numberOfBedrooms']),
                    "Area (sq ft)": round(features['tfarea']),
                    "Current Energy Efficiency": round(features['CURRENT_ENERGY_EFFICIENCY']),
                    "Potential Energy Efficiency": round(features['POTENTIAL_ENERGY_EFFICIENCY']),
                    "Borough": features['borough'],
                    "Property Type": features['house_Type']
                }
                
                # Display features in a nicer format
                col1, col2 = st.columns(2)
                for i, (key, value) in enumerate(feature_display.items()):
                    if i % 2 == 0:
                        col1.metric(key, value)
                    else:
                        col2.metric(key, value)

                # Make prediction
                with st.spinner("Calculating price prediction..."):
                    prediction = predict_house_price_hybrid(
                        ds=datetime.now(),
                        numberrooms=features['numberOfBedrooms'],
                        Postcode=postcode,
                        region=features['borough'] if 'borough' in features else None,
                        house_type=features['house_Type'] if 'house_Type' in features else None,
                        tfarea=features['tfarea'],
                        CURRENT_ENERGY_EFFICIENCY=features['CURRENT_ENERGY_EFFICIENCY'],
                        POTENTIAL_ENERGY_EFFICIENCY=features['POTENTIAL_ENERGY_EFFICIENCY'],
                        postcode_freq_data=postcode_freq_data,
                        pop_all_data=pop_all_data,
                        encoded_features=encoded_features,
                        prophet_model=prophet_model,
                        xgb_res_model=xgb_res_model,
                        property_age="old",  # Default value
                        tenure_type="freehold",  # Default value
                        scale_factors=scale_factors  # Pass scale factors
                    )
                
                # Display the prediction with better formatting
                st.subheader("Price Prediction")
                col1, col2 = st.columns(2)
                col1.metric("Estimated Price", f"£{prediction['Predicted_Price']:,.0f}")
                col2.metric("Price Range", f"£{prediction['Price_Range'][0]:,.0f} - £{prediction['Price_Range'][1]:,.0f}")
                
                # Add a note about the prediction
                st.info("This prediction is based on historical data and market trends. Actual prices may vary based on specific property conditions and market fluctuations.")
            else:
                st.error("No matching data found for the provided inputs. Please check the postcode and try again.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check if all required files (prophet_model.pkl, xgb_res_model.pkl, FinalData.parquet, postcode_freq.csv, scale_factors.pkl) exist in the correct location.")
            
               
# Run the main function
if __name__ == "__main__":
    show()