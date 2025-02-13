import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns  # For better styling
from matplotlib.ticker import FuncFormatter  # For custom y-axis formatting

@st.cache_data  # Cache data to improve performance
def load_price_growth_data(postcode, years_to_load):
    base_path = "split_datasets"  # Unified dataset folder path
    data_list = []

    for year in years_to_load:
        file_path = os.path.join(base_path, f"sales_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=["Date", "Postcode", "Price"])  # Adjust column names
            df["Postcode"] = df["Postcode"].str.strip().str.upper()  # Normalize postcode format
            df = df[df["Postcode"] == postcode.upper()]  # Filter by postcode (case insensitive)
            df["Year"] = pd.to_datetime(df["Date"], errors='coerce').dt.year  # Extract year from date
            data_list.append(df)
        else:
            st.warning(f"Missing data file: {file_path}")

    if data_list:
        full_data = pd.concat(data_list)
        avg_prices = full_data.groupby("Year")["Price"].mean().reset_index()
        return avg_prices
    else:
        return pd.DataFrame({"Year": [], "Price": []})

# Function to format y-axis labels as £
def currency_formatter(x, pos):
    return f"£{x:,.0f}"

def show():
    st.title("Data Analytics")

    # Dropdown for selecting analytics option
    analytics_option = st.selectbox(
        "Choose an option for data analytics:",
        ["House Growth", "Crime Rate", "Average Price"]
    )

    postcode = st.text_input("Enter Postcode for Analysis", max_chars=10)
    
    if analytics_option == "House Growth":
        # Toggle to switch between 5-year and 10-year data
        graph_choice = st.radio("Select Graph Range:", ["Last 5 Years (2020-2024)", "Last 10 Years (2015-2024)"])
        
        if postcode:
            st.subheader(f"House Price Growth in {postcode}")
            
            if graph_choice == "Last 5 Years (2020-2024)":
                df = load_price_growth_data(postcode, range(2020, 2025))
            else:
                df = load_price_growth_data(postcode, range(2015, 2025))
            
            if not df.empty:
                # Set seaborn style for better visuals
                sns.set_style("whitegrid")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df, x="Year", y="Price", marker='o', color='maroon', linewidth=2.5, ax=ax)

                # Set fixed x-axis labels
                ax.set_xticks(df["Year"].unique())
                ax.set_xticklabels(df["Year"].unique(), fontsize=12)

                # Format y-axis labels as £
                ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

                # Add title and labels
                ax.set_title(f"House Price Growth in {postcode}", fontsize=16, fontweight='bold')
                ax.set_xlabel("Year", fontsize=14)
                ax.set_ylabel("Average Price (£)", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)

                # Display the plot
                st.pyplot(fig)
            else:
                st.write("No data available for this postcode.")
    elif analytics_option in ["Crime Rate", "Average Price"]:
        st.subheader(f"{analytics_option} Analysis")
        st.write("Display relevant analytics here.")
