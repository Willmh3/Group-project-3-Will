import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns  # For better styling
from matplotlib.ticker import FuncFormatter  # For custom y-axis formatting

@st.cache_data  # Cache data to improve performance
def load_price_growth_data(postcode):
    base_path = "split_datasets"  # Use relative path for deployment
    years_to_load = [2020, 2021, 2022, 2023, 2024]  # Load years 2020-2024
    data_list = []

    for year in years_to_load:
        file_path = os.path.join(base_path, f"sales_{year}.csv")
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, usecols=["Date", "Postcode", "Price"])  # Adjust column names
                df = df[df["Postcode"] == postcode]  # Filter by postcode
                df["Year"] = pd.to_datetime(df["Date"]).dt.year  # Extract year from date
                data_list.append(df)
        except Exception as e:
            st.error(f"Error loading data for {year}: {e}")

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
        ["Crime Rate", "House Growth", "Average Price"]
    )

    # Display content based on the selected option
    if analytics_option == "Crime Rate":
        st.subheader("Crime Rate Analysis")
        st.write("Display crime rate analysis here.")
        # Add your code for crime rate analysis (e.g., display graphs, statistics, etc.)

    elif analytics_option == "House Growth":
        st.subheader("House Growth Analysis")
        postcode = st.text_input("Enter Postcode for House Growth Analysis", max_chars=10)
        if postcode:
            st.subheader(f"House Price Growth in {postcode} (Last 5 Years)")
            df = load_price_growth_data(postcode)
            
            if not df.empty:
                # Set seaborn style for better visuals
                sns.set_style("whitegrid")

                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df, x="Year", y="Price", marker='o', color='maroon', linewidth=2.5, ax=ax)

                # Set fixed x-axis labels
                ax.set_xticks([2020, 2021, 2022, 2023, 2024])
                ax.set_xticklabels([2020, 2021, 2022, 2023, 2024], fontsize=12)

                # Format y-axis labels as £
                ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

                # Add title and labels
                ax.set_title(f"House Price Growth in {postcode}", fontsize=16, fontweight='bold')
                ax.set_xlabel("Year", fontsize=14)
                ax.set_ylabel("Average Price (£)", fontsize=14)

                # Add grid lines
                ax.grid(True, linestyle='--', alpha=0.7)

                # Annotate the highest and lowest prices
                max_price = df["Price"].max()
                min_price = df["Price"].min()
                ax.annotate(f"Max: £{max_price:,.0f}", 
                            xy=(df.loc[df["Price"].idxmax(), "Year"], max_price),
                            xytext=(10, 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color='black'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))  # Add bounding box
                ax.annotate(f"Min: £{min_price:,.0f}", 
                            xy=(df.loc[df["Price"].idxmin(), "Year"], min_price),
                            xytext=(10, -40), textcoords='offset points',  # Adjusted y-offset
                            arrowprops=dict(arrowstyle="->", color='black'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))  # Add bounding box

                # Display the plot
                st.pyplot(fig)
            else:
                st.write("No data available for this postcode.")

    elif analytics_option == "Average Price":
        st.subheader("Average Price Analysis")
        st.write("Display average price analysis here.")
        # Add your code for average price analysis (e.g., show price comparison graphs, stats)