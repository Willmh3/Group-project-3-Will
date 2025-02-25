import streamlit as st
import pandas as pd
import numpy as np

def calculate_mortgage(principal, interest_rate, loan_term_years):
    """
    Calculate the monthly mortgage payment using the formula for a fixed-rate mortgage.
    """
    # Convert annual interest rate to monthly and loan term to months
    monthly_interest_rate = (interest_rate / 100) / 12
    loan_term_months = loan_term_years * 12

    # Calculate monthly payment
    if monthly_interest_rate == 0:
        monthly_payment = principal / loan_term_months
    else:
        monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months) / ((1 + monthly_interest_rate) ** loan_term_months - 1)

    return monthly_payment

def estimate_future_price(current_price, annual_growth_rate, years):
    """
    Estimate the future price of a house based on an annual growth rate.
    """
    future_price = current_price * (1 + annual_growth_rate / 100) ** years
    return future_price

def show():
    st.title("🏠 Mortgage Calculator")
    st.write("Calculate your monthly mortgage payments and compare them to the estimated future price of a house.")

    # Input fields for mortgage calculation
    st.subheader("Mortgage Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        house_price = st.number_input("House Price (£)", min_value=0, value=300000, step=10000)
    with col2:
        deposit = st.number_input("Deposit (£) (optional)", min_value=0, value=0, step=10000)  # Optional deposit, default to 0
    with col3:
        loan_term_years = st.slider("Loan Term (Years)", min_value=5, max_value=30, value=25, step=1)

    col1, col2 = st.columns(2)
    with col1:
        interest_rate = st.slider("Interest Rate (%)", min_value=0.1, max_value=10.0, value=4.5, step=0.1)
    with col2:
        annual_growth_rate = st.slider("Annual House Price Growth Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

    # Calculate mortgage
    principal = house_price - deposit  # Deposit is optional, default is 0
    monthly_payment = calculate_mortgage(principal, interest_rate, loan_term_years)
    total_payment = monthly_payment * loan_term_years * 12

    # Estimate future house price
    future_price = estimate_future_price(house_price, annual_growth_rate, years=15)

    # Display results with improved styling
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center;'>"
            f"<h3 style='color: #333333;'>Monthly Mortgage Payment</h3>"
            f"<p style='font-size: 24px; font-weight: bold; color: #000000;'>£{monthly_payment:,.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center;'>"
            f"<h3 style='color: #333333;'>Total Mortgage Cost</h3>"
            f"<p style='font-size: 24px; font-weight: bold; color: #000000;'>£{total_payment:,.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.subheader("Future House Price Estimate")
    st.write(f"Based on an annual growth rate of {annual_growth_rate}%, the estimated price of the house in 15 years is:")
    st.markdown(
        f"<div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center;'>"
        f"<p style='font-size: 24px; font-weight: bold; color: #000000;'>£{future_price:,.2f}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Comparison
    st.subheader("Comparison")
    st.write(f"After 15 years, your total mortgage payments will be £{total_payment:,.2f}, and the estimated house price will be £{future_price:,.2f}.")
    if future_price > total_payment:
        st.success("The estimated house price is higher than your total mortgage payments. This is a good investment!")
    else:
        st.warning("The estimated house price is lower than your total mortgage payments. Consider other options.")

if __name__ == "_main_":
    show()