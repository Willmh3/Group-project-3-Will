import streamlit as st
import pandas as pd
import datetime
import math

# Function to determine loan term based on tenure type
def estimate_loan_term(tenure_type):
    if tenure_type == 'F':  # Freehold
        return 30  # Standard mortgage term
    elif tenure_type == 'L':  # Leasehold
        return 50  # Default assumption
    elif tenure_type == 'U':  # Unknown
        return 25  # Default assumption
    else:
        raise ValueError("Invalid tenure type. Use 'L', 'F', or 'U'.")

# Function to calculate mortgage payment with sensitivity analysis
def mortgage_payments(pred_price, loan_term, i, dp=0.015):
    down_payment = pred_price * dp
    loan_amount = pred_price - down_payment
    months = loan_term * 12
    
    # Sensitivity Analysis: ±0.5% around given interest rate
    sensitivity_rates = [i - 0.005, i, i + 0.005]  # ±0.5% variation
    mortgage_payments = {}

    for rate in sensitivity_rates:
        monthly_rate = rate / 12
        M = loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        mortgage_payments[f"{rate*100:.2f}%"] = M

    return mortgage_payments

# Function to calculate mortgage payments for a given date
def projected_mortgage_payment(pred_price, loan_term, interest_rates, target_date):
    today = datetime.date.today()
    target_year = target_date.year
    current_year = today.year
    
    years_diff = max(1, math.ceil(target_year - current_year))  # Round up
    
    # Extend interest_rates list if it's shorter than years_diff
    if years_diff > len(interest_rates):
        last_rate = interest_rates[-1]  # Use the last available rate
        interest_rates.extend([last_rate] * (years_diff - len(interest_rates)))
    
    selected_rate = interest_rates[years_diff - 1]  # Select corresponding interest rate
    results = mortgage_payments(pred_price, loan_term, selected_rate)
    return target_year, results  # Return the prediction year

# Function to create a color-coded table without an index
def display_colored_mortgage_table(prediction_year, results):
    """
    Converts mortgage payments dictionary into a Pandas DataFrame with alternating color formatting.
    """
    table_data = []

    for rate, amount in results.items():
        table_data.append([prediction_year, f"{float(rate.strip('%')):.2f}", f"£{amount:,.2f}"])

    df = pd.DataFrame(table_data, columns=["Year", "Interest Rate (%)", "Monthly Payment (£)"])

    def highlight_cells(row):
        index = row.name  # Get row index
        if index % 3 == 0:
            return ['background-color: lightgreen; color: black'] * len(row)  # Green with black text
        elif index % 3 == 1:
            return ['background-color: white; color: black'] * len(row)  # White with black text
        else:
            return ['background-color: salmon; color: black'] * len(row)  # Red with black text

    styled_df = df.style.apply(highlight_cells, axis=1).hide(axis="index")  # Hide index
    
    return styled_df

# Streamlit UI
def show():
    st.title("Mortgage Calculator")
    
    # Input form
    with st.form("mortgage_form"):
        st.write("Enter the details of the mortgage:")
        
        # Collect user inputs
        house_price = st.number_input("House Price (£)", min_value=0.0, value=320000.0, step=1000.0)
        tenure_type = st.selectbox("Tenure Type", ["Freehold (F)", "Leasehold (L)", "Unknown (U)"], index=0)
        
        # Checkbox to toggle custom loan length input
        use_custom_loan_length = st.checkbox(
            "Input custom loan length (in years)",
            value=False,  # Default to unchecked
            key="use_custom_loan_length"
        )
        
        # Always display the loan length input box
        loan_length = st.number_input(
            "Desired Loan Length (years)",
            min_value=1,
            max_value=50,
            value=25,  # Default value
            step=1,
            disabled=not use_custom_loan_length  # Disable if checkbox is unchecked
        )
        
        # If checkbox is unchecked, calculate loan length based on tenure type
        if not use_custom_loan_length:
            tenure_code = tenure_type.split(" ")[1].strip("()")
            loan_length = estimate_loan_term(tenure_code)
        
        # Example interest rates (can be replaced with dynamic data)
        interest_rates = [0.04344, 0.04252, 0.03973, 0.04166, 0.0428, 0.0418, 0.04325, 0.0442, 0.049, 0.0493, 0.0493, 0.0468, 0.0468, 0.0468, 0.04898]
        
        submitted = st.form_submit_button("Calculate Mortgage Payments")

        if submitted:
            try:
                # Calculate target date based on loan length
                today = datetime.date.today()
                target_date = today + datetime.timedelta(days=loan_length * 365)
                
                # Run the function to get mortgage payments
                prediction_year, results = projected_mortgage_payment(house_price, loan_length, interest_rates, target_date)
                
                # Display the color-coded mortgage table
                st.subheader("Mortgage Payment Projections")
                styled_table = display_colored_mortgage_table(prediction_year, results)
                st.dataframe(styled_table, use_container_width=True)
            
            except ValueError as e:
                st.error(f"Error: {str(e)}")

# Run the main function
if __name__ == "__main__":
    show()