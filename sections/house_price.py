import streamlit as st
import pandas as pd
import re

def is_valid_uk_postcode(postcode):
    """
    Validates a UK postcode using a regular expression.
    """
    # UK postcode regex pattern
    pattern = r"^([A-Za-z]{1,2}\d{1,2}[A-Za-z]? \d[A-Za-z]{2})$"
    return re.match(pattern, postcode) is not None

def show():
    st.title("House Price Prediction")

    # Input fields with smaller box sizes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        house_number = st.text_input("House Number (optional)", key="house_number", max_chars=10)
    with col2:
        road = st.text_input("Road (optional)", key="road", max_chars=30)
    with col3:
        postcode = st.text_input("Postcode", key="postcode", max_chars=10)
    with col4:
        flat_number = st.text_input("Flat Number (optional)", key="flat_number", max_chars=10)

    # Initialize session state variables if they don't exist
    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None

    # Function to update predicted price based on inputs
    def update_predicted_price():
        if postcode and is_valid_uk_postcode(postcode):  # Use custom validation function
            st.session_state.predicted_price = "$350,000"  # Placeholder (replace with dummy logic)
        else:
            st.session_state.predicted_price = "Invalid or missing postcode. Please enter a valid UK postcode."

    update_predicted_price()

    # Display predicted price
    st.subheader("Predicted House Price")
    price_box = f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px;">
        {st.session_state.predicted_price}
    </div>
    """
    st.markdown(price_box, unsafe_allow_html=True)