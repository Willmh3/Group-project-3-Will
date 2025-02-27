import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
#Load the required data

postcode_freq = pd.read_csv('postcode_freq.csv')
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

# Merge df_all with df4 to get the required columns
df_all = df_all.merge(df4[['Unique_Reference', 'tfarea', 'numberrooms', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY']],
                     on='Unique_Reference', how='left')

# Drop unnecessary columns and rows with missing values
df_all.drop(columns=["ID", "Unique_Reference", "House_Number", "Flat_Number", "A", "A.1"], inplace=True)
df_all.dropna(subset=['Date', 'Price', 'tfarea', 'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY'], inplace=True)

# Convert 'Date' to datetime and sort by date
df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y-%m-%d %H:%M', errors='coerce')
df_all.dropna(subset=['Date'], inplace=True)
df_all.sort_values(by='Date', inplace=True)
df_all.reset_index(drop=True, inplace=True)

# --- Incorporate Non-Numeric Data ---
df_all['postcode_freq'] = df_all['Postcode'].map(df_all['Postcode'].value_counts(normalize=True))
df_all = pd.get_dummies(df_all, columns=['Region', 'Tenure_Type', 'House_Type'], drop_first=True)

# Remove outliers by year
df_all['Year'] = df_all['Date'].dt.year
def remove_outliers(group):
    Q1 = group['Price'].quantile(0.25)
    Q3 = group['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group['Price'] >= lower_bound) & (group['Price'] <= upper_bound)]

# Fix for older versions of pandas
df_all = df_all.groupby('Year', group_keys=False).apply(lambda group: remove_outliers(group.drop(columns=['Year'])))


# ===============================
# Monthly Aggregation & Time Features
# ===============================
df_all.set_index('Date', inplace=True)

# Fix for FutureWarning: resample('M') -> resample('ME')
monthly_data = df_all.resample('ME').mean(numeric_only=True).reset_index()
monthly_data['Year'] = monthly_data['Date'].dt.year
monthly_data['Month'] = monthly_data['Date'].dt.month
monthly_data['Year_offset'] = monthly_data['Year'] - 1995
monthly_data['Year_offset_sq'] = monthly_data['Year_offset'] ** 2
# Load the trained models
prophet_model = joblib.load('prophet_model.pkl')
xgb_res_model = joblib.load('xgb_res_model.pkl')

def predict_house_price_hybrid(ds, tfarea, numberrooms, CURRENT_ENERGY_EFFICIENCY, POTENTIAL_ENERGY_EFFICIENCY,
                              Postcode, region, tenure_type, house_type, alpha=1.0):
    """
    Predicts house price using the hybrid model (Prophet base + alpha*XGBoost residual correction).
    """
    import pandas as pd

    # Convert ds to datetime and compute time features
    ds = pd.to_datetime(ds)
    Year = ds.year
    Month = ds.month
    Year_offset = Year - 1995
    Year_offset_sq = Year_offset ** 2

    # Retrieve defaults from monthly_data
    defaults = monthly_data.mean()
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
        'postcode_freq': df_all['Postcode'].value_counts(normalize=True).get(Postcode, 0.0),
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
    final_feature_cols = list(monthly_data.drop(columns=["Date", "Price"]).columns)
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

def show():
    st.title("House Price Prediction")

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Postcode")
        postcode = st.text_input("Enter Postcode (e.g., SW1A 1AA)", key="postcode", max_chars=10)
    with col2:
        st.subheader("Borough")
        boroughs = [
            "Barking And Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "Croydon", "Ealing", "Enfield", 
            "Greenwich", "Hackney", "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington And Chelsea", 
            "Kingston Upon Thames", "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", "Richmond Upon Thames", 
            "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
        ]
        boroughs.sort()
        selected_borough = st.selectbox("Select a Borough", ["Unknown"] + boroughs, index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("House Type (Optional)")
        house_type = st.selectbox("Select the house type", ["Unknown", "Semi-Detached", "Detached", "Terraced", "Flat"], index=0)
    with col2:
        st.subheader("Bedrooms (Optional)")
        bedroom_toggle = st.selectbox("Select the number of bedrooms", ["Unknown"] + [str(i) for i in range(1, 16)], index=0)

    # Prediction button
    predict_button = st.button("Predict Price", key="predict_button", use_container_width=True)

    if predict_button:
        if not postcode:
            st.error("Please enter a postcode.")
        else:
            # Prepare inputs for the prediction function
            inputs = {
                "ds": datetime.now().strftime("%Y-%m-%d"),  # Current date
                "tfarea": 100.0,  # Default value (can be adjusted)
                "numberrooms": int(bedroom_toggle) if bedroom_toggle != "Unknown" else None,
                "CURRENT_ENERGY_EFFICIENCY": 80,  # Default value
                "POTENTIAL_ENERGY_EFFICIENCY": 90,  # Default value
                "Postcode": postcode,
                "region": selected_borough if selected_borough != "Unknown" else None,
                "tenure_type": "Freehold",  # Default to Freehold
                "house_type": house_type if house_type != "Unknown" else None,
                "alpha": 0.7  # Default weight for residual correction
            }

            # Predict current price
            current_prediction = predict_house_price_hybrid(**inputs)

            # Predict future price (15 years from now)
            future_date = (datetime.now() + timedelta(days=365 * 15)).strftime("%Y-%m-%d")
            inputs["ds"] = future_date
            future_prediction = predict_house_price_hybrid(**inputs)

            # Display current price prediction
            st.subheader("Current Price Prediction")
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                f"<h3>Predicted House Price</h3>"
                f"<p style='font-size: 24px; font-weight: bold;'>£{current_prediction['Predicted_Price']:,.2f}</p>"
                f"<p style='font-size: 16px;'>Price Range: £{current_prediction['Price_Range'][0]:,.2f} - £{current_prediction['Price_Range'][1]:,.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Display 15-year future estimate
            st.subheader("15-Year Future Estimate")
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                f"<h3>Predicted House Price in 15 Years</h3>"
                f"<p style='font-size: 24px; font-weight: bold;'>£{future_prediction['Predicted_Price']:,.2f}</p>"
                f"<p style='font-size: 16px;'>Price Range: £{future_prediction['Price_Range'][0]:,.2f} - £{future_prediction['Price_Range'][1]:,.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    show()