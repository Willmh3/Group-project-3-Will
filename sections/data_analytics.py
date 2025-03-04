import streamlit as st
import pandas as pd
import plotly.express as px
import re
from functools import lru_cache
import glob

@lru_cache(maxsize=1)
def load_data():
    """
    Load and preprocess housing data from Parquet file.
    
    Returns:
        pd.DataFrame: Processed housing data
    """
    try:
        # Select only the necessary columns
        columns_to_load = [
            'price', 'dateOfTransfer', 'postcode', 
            'house_Type', 'borough', 'numberOfBedrooms','House_Number','Street_Name'
        ]
        
        # Load data
        df = pd.read_parquet('FinalData.parquet', columns=columns_to_load)
        
        # Data cleaning and transformation
        df['postcode'] = df['postcode'].astype(str).str.upper()
        df['borough'] = df['borough'].str.title()
        df['Year'] = pd.to_datetime(df['dateOfTransfer'], errors='coerce').dt.year
        
        # House type mapping
        house_type_mapping = {
            'F': 'Flat',
            'House': 'House',
            'D': 'Detached',
            'S': 'Semi-Detached',
            'T': 'Terraced'
        }
        df['house_Type'] = df['house_Type'].map(house_type_mapping).fillna('Other')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def is_valid_postcode(postcode):
    """Basic UK postcode validation."""
    pattern = r"^[A-Z]{1,2}[0-9R][0-9A-Z]? [0-9][A-Z]{2}$"
    return re.match(pattern, postcode) is not None

def show():
    """Main function to display the analytics dashboard."""
    st.title("📊 Housing Data Analytics Dashboard")

    # Prepare filter options
    all_boroughs = sorted(load_data()['borough'].unique().tolist())
    all_house_types = sorted(load_data()['house_Type'].unique().tolist())

    # Filters
    st.header("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        boroughs = ["All"] + all_boroughs
        selected_borough = st.selectbox("🏙 Select Borough", boroughs, key="borough_select")
    
    with col2:
        house_types = ["All"] + all_house_types
        selected_house_type = st.selectbox("🏠 Select House Type", house_types, key="house_type_select")

    with col3:
        postcode = st.text_input("📍 Enter Postcode", placeholder="e.g., SW1A 1AA", key="postcode_input").strip().upper()

    # Load and filter data when button is pressed
    if st.button("Apply Filters", type="primary"):
        with st.spinner("Loading data..."):
            full_data = load_data()
            
            # Apply filters
            filtered_data = full_data.copy()
            
            if selected_borough != "All":
                filtered_data = filtered_data[filtered_data["borough"] == selected_borough]
            
            if selected_house_type != "All":
                filtered_data = filtered_data[filtered_data["house_Type"] == selected_house_type]
            
            if postcode and is_valid_postcode(postcode):
                filtered_data = filtered_data[filtered_data["postcode"] == postcode]

            # Key Metrics
            st.header("Key Metrics")
            col1, col2 = st.columns(2)

            with col1:
                num_houses = filtered_data.shape[0]
                st.metric(label="📊 Data Points", value=f"{num_houses:,}")

                lower_quartile = filtered_data["price"].quantile(0.25) if not filtered_data.empty else 0
                st.metric(label="🔻 Lower Quartile Price", value=f"£{lower_quartile:,.0f}")
            
                avg_price = filtered_data["price"].mean() if not filtered_data.empty else 0
                st.metric(label="🏠 Mean House Price", value=f"£{avg_price:,.0f}")
            
                upper_quartile = filtered_data["price"].quantile(0.75) if not filtered_data.empty else 0
                st.metric(label="🔺 Upper Quartile Price", value=f"£{upper_quartile:,.0f}")

            with col2:
                if not filtered_data.empty:
                    house_type_counts = filtered_data["house_Type"].value_counts()
                    fig = px.pie(house_type_counts, names=house_type_counts.index, values=house_type_counts.values, title="House Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data found matching the selected filters.")

            # Liquidity Plot (Only if a postcode is entered)
            if postcode and is_valid_postcode(postcode):
                st.header("📉 Liquidity Over Time")
                liquidity_data = filtered_data.groupby("Year")["postcode"].count().reset_index()
                liquidity_data.rename(columns={"postcode": "Houses Sold"}, inplace=True)

                liquidity_data = liquidity_data[(liquidity_data["Year"] >= 2005) & (liquidity_data["Year"] <= 2024)]

                if not liquidity_data.empty:
                    fig_liquidity = px.line(liquidity_data, x="Year", y="Houses Sold", title="Houses Sold Per Year (Liquidity)")
                    fig_liquidity.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_liquidity, use_container_width=True)
                else:
                    st.warning("No sales data available for this postcode between 2005-2024.")

            # Bedroom Distribution
            st.header("Bedroom Distribution")
            bedroom_counts = filtered_data["numberOfBedrooms"].value_counts().sort_index()
            fig_bedrooms = px.bar(x=bedroom_counts.index, y=bedroom_counts.values, 
                                   title="Number of Bedrooms Distribution",
                                   labels={'x': 'Number of Bedrooms', 'y': 'Count'})
            st.plotly_chart(fig_bedrooms, use_container_width=True)

            # Download Button
            st.header("Download Filtered Data")
            csv_data = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download CSV", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("📍 *Note:* This dashboard uses data from FinalData.parquet")

# For direct script execution
if __name__ == "__main__":
    show()