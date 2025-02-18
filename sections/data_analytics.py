import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import re

@st.cache_data
def load_price_growth_data(postcode, years_to_load):
    base_path = "split_datasets"
    data_list = []

    for year in years_to_load:
        file_path = os.path.join(base_path, f"sales_{year}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=["Date", "Postcode", "Price"])
            df["Postcode"] = df["Postcode"].str.strip().str.upper()
            df = df[df["Postcode"] == postcode.upper()]
            df["Year"] = pd.to_datetime(df["Date"], errors='coerce').dt.year
            data_list.append(df)
        else:
            st.warning(f"Missing data file: {file_path}")

    if data_list:
        full_data = pd.concat(data_list)
        if not full_data.empty:
            avg_prices = full_data.groupby("Year")["Price"].mean().reset_index()
            return avg_prices
        else:
            st.warning("No data available for the selected years.")
            return pd.DataFrame({"Year": [], "Price": []})
    else:
        st.warning("No data files were found for the selected years.")
        return pd.DataFrame({"Year": [], "Price": []})

def currency_formatter(x, pos):
    return f"£{x:,.0f}"

def is_valid_uk_postcode(postcode):
    pattern = r"^([A-Za-z]{1,2}\d{1,2}[A-Za-z]? \d[A-Za-z]{2})$"
    return re.match(pattern, postcode) is not None

def show():
    st.title("Data Analytics")

    analytics_option = st.selectbox(
        "Choose an option for data analytics:",
        ["House Growth", "Crime Rate", "Average Price"]
    )

    postcode = st.text_input("Enter Postcode for Analysis", max_chars=10)

    if analytics_option == "House Growth":
        graph_range = st.selectbox("Select Graph Range:", ["Last 5 Years (2020-2024)", "Last 10 Years (2015-2024)"])
        
        if postcode:
            if not is_valid_uk_postcode(postcode):
                st.error("Please enter a valid UK postcode.")
            else:
                with st.spinner("Loading data..."):
                    if graph_range == "Last 5 Years (2020-2024)":
                        df = load_price_growth_data(postcode, range(2020, 2025))
                    else:
                        df = load_price_growth_data(postcode, range(2015, 2025))

                if not df.empty:
                    sns.set_style("whitegrid")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=df, x="Year", y="Price", marker='o', color='maroon', linewidth=2.5, ax=ax)

                    ax.set_xticks(df["Year"].unique())
                    ax.set_xticklabels(df["Year"].unique(), fontsize=12)
                    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

                    ax.set_title(f"House Price Growth in {postcode}", fontsize=16, fontweight='bold')
                    ax.set_xlabel("Year", fontsize=14)
                    ax.set_ylabel("Average Price (£)", fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)

                    max_price = df["Price"].max()
                    min_price = df["Price"].min()
                    ax.annotate(f"Max: £{max_price:,.0f}", 
                                xy=(df.loc[df["Price"].idxmax(), "Year"], max_price),
                                xytext=(10, 20), textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", color='black'),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
                    ax.annotate(f"Min: £{min_price:,.0f}", 
                                xy=(df.loc[df["Price"].idxmin(), "Year"], min_price),
                                xytext=(10, -40), textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", color='black'),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for postcode {postcode}. Please check the postcode or try another one.")
    elif analytics_option in ["Crime Rate", "Average Price"]:
        st.subheader(f"{analytics_option} Analysis")
        st.write("Display relevant analytics here.")