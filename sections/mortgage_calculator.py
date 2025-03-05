import streamlit as st
import pandas as pd
import datetime
import math
import plotly.graph_objs as plt

# Import the necessary functions from the house price prediction script
from sections.house_price import extract_features, predict_house_price_hybrid, load_data, load_models, load_population_data, load_postcode_freq, load_encoded_features, load_scale_factors

def estimate_loan_term(tenure_type):
    """Recommend loan term based on property tenure type."""
    tenure_terms = {
        'F': 30,   # Freehold: standard 30-year mortgage
        'L': 25,   # Leasehold: slightly shorter term
        'U': 25    # Unknown: conservative estimate
    }
    return tenure_terms.get(tenure_type, 25)

def mortgage_payments(pred_price, interest_rates, dp=0.10, dp_variation=0.015):
    """
    Calculate mortgage payments with sensitivity analysis using predefined interest rates.
    
    Args:
    - pred_price: Property price
    - interest_rates: Predefined list of interest rates
    - dp: Deposit percentage (default 10%)
    - dp_variation: Deposit percentage variation
    
    Returns:
    Dictionary of mortgage payments with different scenarios
    """
    # Use the first (current) interest rate from the list
    selected_rate = interest_rates[0]
    
    down_payment = pred_price * dp
    loan_amount = pred_price - down_payment
    loan_term = 25  # Fixed loan term
    months = loan_term * 12
    
    # Sensitivity Analysis: ±0.5% around selected interest rate
    sensitivity_rates = [selected_rate - 0.005, selected_rate, selected_rate + 0.005]
    mortgage_payments = {}

    for rate in sensitivity_rates:
        monthly_rate = rate / 12
        M = loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        mortgage_payments[f"{rate*100:.2f}%"] = {
            'monthly_payment': M,
            'down_payment': down_payment,
            'total_paid': M * months,
            'total_interest': (M * months) - loan_amount
        }

    return mortgage_payments

def show():
    st.title("🏡 Comprehensive Mortgage Calculator")
    
    # Predefined interest rates (matching previous implementation)
    interest_rates = [0.04344, 0.04252, 0.03973, 0.04166, 0.0428, 0.0418, 0.04325, 0.0442, 0.049, 0.0493, 0.0493, 0.0468, 0.0468, 0.0468, 0.04898]
    
    # Load necessary models and data
    with st.spinner("Loading prediction models..."):
        prophet_model, xgb_res_model = load_models()
        df = load_data()
        pop_all_data = load_population_data()
        postcode_freq_data = load_postcode_freq()
        encoded_features = load_encoded_features()
        scale_factors = load_scale_factors()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price & Mortgage", "💡 Insights", "🧮 Advanced Options"])
    
    with tab1:
        # DEPOSIT DISCLAIMER
        st.warning("""
        🚨 Important Deposit Notice 🚨
        - This calculator uses a FIXED 10% deposit
        - Actual deposit requirements vary by lender
        - Some mortgages may require 15-20% deposit
        - Lower deposits might incur higher interest rates
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price Determination Section
            st.subheader("🏘️ Property Valuation")
            price_method = st.radio(
                "How would you like to determine the house price?", 
                ["Predict Price", "Enter Manually"],
                help="Choose whether to use our prediction model or enter a price manually."
            )
            
            # Prediction or Manual Input
            if price_method == "Predict Price":
                with st.expander("Property Details for Prediction"):
                    postcode = st.text_input("Postcode").upper()
                    street_name = st.text_input("Street Name (optional)")
                    house_number = st.text_input("House Number (optional)")
            else:
                predicted_price = st.number_input(
                    "House Price (£)", 
                    min_value=50000.0, 
                    max_value=5000000.0, 
                    value=500000.0, 
                    step=50000.0,
                    help="Enter the total value of the property you're interested in."
                )
            
            # Mortgage Details
            st.subheader("💰 Mortgage Details")
            
            # Loan Length Selection
            loan_length = st.select_slider(
                "Loan Term (Years)", 
                options=[10, 15, 20, 25, 30, 35, 40],
                value=25,
                help="Duration over which you'll repay the mortgage"
            )
            
            # Calculate Button
            if st.button("Calculate Mortgage Details", type="primary"):
                try:
                    if price_method == "Predict Price":
                        # Validate postcode input
                        if not postcode or postcode.strip() == "":
                            st.error("Please enter a postcode for price prediction.")
                            st.stop()
                        
                        # Attempt to extract features
                        features = extract_features(df, postcode, street_name, house_number)
                        
                        if features is None:
                            st.error("No matching data found for the given postcode. Please try a different postcode or enter the price manually.")
                            st.stop()
                        
                        # Predict house price
                        prediction = predict_house_price_hybrid(
                            ds=datetime.datetime.now(),
                            numberrooms=features['numberOfBedrooms'],
                            Postcode=postcode,
                            region=features['borough'],
                            house_type=features['house_Type'],
                            tfarea=features['tfarea'],
                            CURRENT_ENERGY_EFFICIENCY=features['CURRENT_ENERGY_EFFICIENCY'],
                            POTENTIAL_ENERGY_EFFICIENCY=features['POTENTIAL_ENERGY_EFFICIENCY'],
                            postcode_freq_data=postcode_freq_data,
                            pop_all_data=pop_all_data,
                            encoded_features=encoded_features,
                            prophet_model=prophet_model,
                            xgb_res_model=xgb_res_model,
                            property_age="old",
                            tenure_type="freehold",
                            scale_factors=scale_factors
                        )
                        
                        predicted_price = prediction['Predicted_Price']
                        st.success(f"Predicted House Price: £{predicted_price:,.0f}")
                    
                    # Calculate mortgage payments with error bounds
                    results = mortgage_payments(
                        predicted_price, 
                        interest_rates
                    )
                    
                    # Display results
                    st.subheader("📈 Mortgage Payment Scenarios")
                    
                    # Create DataFrame for display
                    table_data = []
                    for rate, details in results.items():
                        table_data.append([
                            rate, 
                            f"£{details['monthly_payment']:,.2f}", 
                            f"£{details['down_payment']:,.2f}",
                            f"£{details['total_paid']:,.2f}",
                            f"£{details['total_interest']:,.2f}"
                        ])
                    
                    df = pd.DataFrame(table_data, columns=[
                        "Interest Rate", 
                        "Monthly Payment", 
                        "Down Payment", 
                        "Total Paid", 
                        "Total Interest"
                    ])
                    
                    # Color-styled table with alternating colors
                    def color_rows(row):
                        index = row.name % 3
                        if index == 0:
                            return ['background-color: lightgreen'] * len(row)
                        elif index == 1:
                            return ['background-color: white'] * len(row)
                        else:
                            return ['background-color: salmon'] * len(row)
                    
                    styled_df = df.style.apply(color_rows, axis=1)
                    
                    # Display styled DataFrame
                    st.dataframe(styled_df)
                    
                    # Detailed explanation
                    st.info("""
                    ### Mortgage Payment Analysis
                    - Scenarios show monthly payments at ±0.5% interest rate variation
                    - Green: Lower end of rate range
                    - White: Mid-point rate
                    - Red: Upper end of rate range
                    - Interest rates used are predefined historical/projected rates
                    """)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        with col2:
            # Intentionally left empty
            pass
    
    with tab2:
        st.header("📊 Mortgage Insights")
        st.write("""
        ### Understanding Your Mortgage
        
        **What impacts your mortgage?**
        - Property Value
        - Deposit Amount (Currently Fixed at 10%)
        - Interest Rates
        - Loan Duration
        
        **Strategies to Reduce Mortgage Costs:**
        1. Save for a larger deposit
        2. Improve credit score
        3. Choose shorter loan terms
        4. Shop around for best rates
        """)
    
    with tab3:
        st.header("🧮 Advanced Options")
        st.write("Coming soon: Detailed amortization schedules and advanced financial modeling.")

# Run the main function
if __name__ == "__main__":
    show()