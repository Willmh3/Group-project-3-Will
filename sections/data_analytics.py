import streamlit as st
import pandas as pd
import plotly.express as px
import re
from functools import lru_cache
import glob

# List of valid London boroughs
valid_london_boroughs = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", 
    "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham", 
    "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", 
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", 
    "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", 
    "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
]

# Mapping for house types
house_type_mapping = {
    "F": "Flat",
    "D": "Detached",
    "T": "Terraced",
    "S": "Semi-Detached",
    "O": "Other"
}

@lru_cache(maxsize=1)
def load_data():
    """
    Load and preprocess housing data with caching to improve performance.
    """
    try:
        # Simulate loading multiple CSV files (replace with actual data loading)
        csv_files = sorted(glob.glob("split_*.csv"))
        if not csv_files:
            st.error("No CSV files found.")
            return pd.DataFrame()
        
        df_list = [pd.read_csv(file) for file in csv_files]
        full_data = pd.concat(df_list, ignore_index=True)

        # Rename columns
        full_data = full_data.rename(columns={
            "NEWCASTLE.UPON.TYNE.1": "Region",
            "S": "House_Type",
            "X42000": "Price",
            "NE4.9DN": "Postcode",
            "Date": "Date"
        })

        # Convert and clean data
        full_data["Postcode"] = full_data["Postcode"].astype(str).str.upper()
        full_data["Region"] = full_data["Region"].str.title()
        full_data["Price"] = pd.to_numeric(full_data["Price"], errors="coerce")
        full_data["House_Type"] = full_data["House_Type"].map(house_type_mapping).fillna("Unknown")

        # Convert date column to datetime format
        full_data["Year"] = pd.to_datetime(full_data["Date"], errors="coerce").dt.year
        
        return full_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def is_valid_postcode(postcode):
    """Basic UK postcode validation."""
    pattern = r"^[A-Z]{1,2}[0-9R][0-9A-Z]? [0-9][A-Z]{2}$"
    return re.match(pattern, postcode) is not None

def show():
    """Main function to display the analytics dashboard."""
    st.title("📊 Data Analytics Dashboard")

    # Filters
    st.header("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        boroughs = ["All"] + sorted(valid_london_boroughs)
        selected_borough = st.selectbox("🏙 Select Borough", boroughs, key="borough_select")
    
    with col2:
        house_types = ["All"] + list(house_type_mapping.values())
        selected_house_type = st.selectbox("🏠 Select House Type", house_types, key="house_type_select")

    with col3:
        postcode = st.text_input("📍 Enter Postcode", placeholder="e.g., SW1A 1AA", key="postcode_input").strip().upper()

    # Load data when button is pressed
    if st.button("Apply Filters", type="primary"):
        with st.spinner("Loading data..."):
            full_data = load_data()
            
            # Apply filters
            filtered_data = full_data.copy()
            
            if selected_borough != "All":
                filtered_data = filtered_data[filtered_data["Region"] == selected_borough]
            
            if selected_house_type != "All":
                filtered_data = filtered_data[filtered_data["House_Type"] == selected_house_type]
            
            if postcode and is_valid_postcode(postcode):
                filtered_data = filtered_data[filtered_data["Postcode"] == postcode]

            # Key Metrics
            st.header("Key Metrics")
            col1, col2 = st.columns(2)

            with col1:
                num_houses = filtered_data.shape[0]
                st.metric(label="📊 Data Points", value=f"{num_houses:,}")

                lower_quartile = filtered_data["Price"].quantile(0.25) if not filtered_data.empty else 0
                st.metric(label="🔻 Lower Quartile Price", value=f"£{lower_quartile:,.0f}")
            
                avg_price = filtered_data["Price"].mean() if not filtered_data.empty else 0
                st.metric(label="🏠 Mean House Price", value=f"£{avg_price:,.0f}")
            
                upper_quartile = filtered_data["Price"].quantile(0.75) if not filtered_data.empty else 0
                st.metric(label="🔺 Upper Quartile Price", value=f"£{upper_quartile:,.0f}")

            with col2:
                if not filtered_data.empty:
                    house_type_counts = filtered_data["House_Type"].value_counts()
                    fig = px.pie(house_type_counts, names=house_type_counts.index, values=house_type_counts.values, title="House Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data found matching the selected filters.")

            # Liquidity Plot (Only if a postcode is entered)
            if postcode and is_valid_postcode(postcode):
                st.header("📉 Liquidity Over Time")
                liquidity_data = filtered_data.groupby("Year")["Postcode"].count().reset_index()
                liquidity_data.rename(columns={"Postcode": "Houses Sold"}, inplace=True)

                liquidity_data = liquidity_data[(liquidity_data["Year"] >= 2005) & (liquidity_data["Year"] <= 2024)]

                if not liquidity_data.empty:
                    fig_liquidity = px.line(liquidity_data, x="Year", y="Houses Sold", title="Houses Sold Per Year (Liquidity)")
                    fig_liquidity.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_liquidity, use_container_width=True)
                else:
                    st.warning("No sales data available for this postcode between 2005-2024.")

            # Download Button
            st.header("Download Filtered Data")
            csv_data = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download CSV", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("📍 *Note:* This dashboard uses real data from split_*.csv.")

# For direct script execution
if __name__ == "__main__":
    show()