import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import glob

# Load the saved models
prophet_model = joblib.load('prophet_model.pkl')
xgb_res_model = joblib.load('xgb_res_model.pkl')

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
pop_all = pd.concat([pop_all, pop_future], ignore_index=True)

# Merge population data with monthly_data
monthly_data = pd.merge(monthly_data, pop_all, on='Year', how='left')

# Load the postcode frequency data (assuming it's saved as a CSV)
postcode_freq = pd.read_csv('postcode_freq.csv')

# Define the prediction function
def predict_house_price_hybrid(ds, numberrooms, Postcode, region, house_type, alpha=1.0):
    """
    Predicts house price using the hybrid model (Prophet base + alpha*XGBoost residual correction).
    Only uses postcode, borough (region), house type, and number of rooms.
    """
    import pandas as pd

    # Convert ds to datetime and compute time features
    ds = pd.to_datetime(ds)
    Year = ds.year
    Month = ds.month
    Year_offset = Year - 1995
    Year_offset_sq = Year_offset ** 2

    # Retrieve defaults for other features (not provided by the user)
    defaults = {
        'tfarea': 100.0,  # Default total floor area
        'CURRENT_ENERGY_EFFICIENCY': 60.0,  # Default energy efficiency
        'POTENTIAL_ENERGY_EFFICIENCY': 70.0,  # Default potential energy efficiency
    }

    # Prepare input values
    input_vals = {
        'ds': ds,
        'Year': Year,
        'Month': Month,
        'Year_offset': Year_offset,
        'Year_offset_sq': Year_offset_sq,
        'numberrooms': numberrooms,
        'tfarea': defaults['tfarea'],
        'CURRENT_ENERGY_EFFICIENCY': defaults['CURRENT_ENERGY_EFFICIENCY'],
        'POTENTIAL_ENERGY_EFFICIENCY': defaults['POTENTIAL_ENERGY_EFFICIENCY'],
        'postcode_freq': postcode_freq[postcode_freq['Postcode'] == Postcode]['Frequency'].values[0] if Postcode in postcode_freq['Postcode'].values else postcode_freq['Frequency'].mean(),
        'Population': pop_all.loc[pop_all['Year'] == Year, 'Population'].values[0]
    }

    # Add encoded features for region, tenure type, and house type
    encoded_features = [col for col in monthly_data.columns if col.startswith('Region_') or 
                        col.startswith('Tenure_Type_') or col.startswith('House_Type_')]
    encoded_vals = {feat: 0.0 for feat in encoded_features}

    # Update for region
    if region is not None:
        desired_region = region.upper()
        for feat in encoded_features:
            if feat.startswith("Region_"):
                encoded_vals[feat] = 1.0 if feat.split("Region_")[1].upper() == desired_region else 0.0

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

# Streamlit UI
def show():
    st.title("House Price Prediction")

    # Input form
    with st.form("house_price_form"):
        st.write("Enter the details of the house:")
        
        # Collect only the required inputs
        postcode = st.text_input("Postcode")
        region = st.text_input("Borough")
        house_type = st.text_input("House Type")
        numberrooms = st.number_input("Number of Rooms", min_value=1, max_value=10)
        
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Use the current date for prediction
        date = datetime.now()

        # Call the prediction function
        prediction = predict_house_price_hybrid(
            ds=date,
            numberrooms=numberrooms,
            Postcode=postcode,
            region=region,
            house_type=house_type
        )
        
        # Display the prediction
        st.write(f"Predicted Price: £{prediction['Predicted_Price']:,.2f}")
        st.write(f"Price Range: £{prediction['Price_Range'][0]:,.2f} - £{prediction['Price_Range'][1]:,.2f}")

# Run the main function
if __name__ == "__main__":
    show()