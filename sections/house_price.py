import streamlit as st
import folium
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

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

    # Function to validate London postcode
    def is_valid_london_postcode(postcode):
        london_postcode_regex = re.compile(r"^(([A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2})|GIR 0AA)$", re.IGNORECASE)
        return london_postcode_regex.match(postcode) is not None

    # Function to update predicted price based on inputs
    def update_predicted_price():
        if postcode and is_valid_london_postcode(postcode):
            st.session_state.predicted_price = "$350,000"  # Placeholder
        else:
            st.session_state.predicted_price = "Invalid or missing postcode. Please enter a valid London postcode."

    update_predicted_price()

    # Display predicted price
    st.subheader("Predicted House Price")
    price_box = f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px;">
        {st.session_state.predicted_price}
    </div>
    """
    st.markdown(price_box, unsafe_allow_html=True)

    # Function to load and aggregate house price data efficiently
    def load_price_growth_data(postcode):
        base_path = r"/workspaces/Group-project-3-Will/split_datasets"  # Adjust with actual path
        years_to_load = [2020, 2021, 2022, 2023, 2024]  # Load years 2020-2024
        data_list = []

        for year in years_to_load:
            file_path = os.path.join(base_path, f"sales_{year}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, usecols=["Date", "Postcode", "Price"])  # Adjust column names
                df = df[df["Postcode"] == postcode]  # Filter by postcode
                df["Year"] = pd.to_datetime(df["Date"]).dt.year  # Extract year from date
                data_list.append(df)

        if data_list:
            full_data = pd.concat(data_list)
            avg_prices = full_data.groupby("Year")["Price"].mean().reset_index()
            return avg_prices
        else:
            return pd.DataFrame({"Year": [], "Price": []})

    # Show price growth trend
    if postcode and is_valid_london_postcode(postcode):
        st.subheader("House Price Growth (Last 5 Years)")
        df = load_price_growth_data(postcode)
        
        if not df.empty:
            fig, ax = plt.subplots()
            ax.plot(df["Year"], df["Price"], marker='o', linestyle='-')
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Price (£)")
            ax.set_title(f"Price Growth in {postcode}")
            st.pyplot(fig)
        else:
            st.write("No data available for this postcode.")
