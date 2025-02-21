import streamlit as st
import pandas as pd
import re
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def is_valid_uk_postcode(postcode):
    """Validates a UK postcode using a regular expression."""
    pattern = r"^([A-Za-z]{1,2}\d{1,2}[A-Za-z]? \d[A-Za-z]{2})$"
    return re.match(pattern, postcode) is not None

def get_coordinates(postcode):
    """Uses geopy to get latitude and longitude for a given postcode."""
    geolocator = Nominatim(user_agent="house_price_app")
    location = geolocator.geocode(postcode)
    if location:
        return location.latitude, location.longitude
    return None, None

def search_house_details(postcode, house_number, road, flat_number=None):
    """Searches for house details in the split CSV files based on postcode, house number, road, and flat number."""
    split_files = [f"split_{i}.csv" for i in range(1, 128)]  # Files split_1.csv to split_127.csv

    for file in split_files:
        try:
            df = pd.read_csv(file)

            # Apply filtering based on available inputs
            condition = df.iloc[:, 4].str.contains(postcode, case=False, na=False)  # Postcode (5th column)
            if house_number:
                condition &= df.iloc[:, 8].str.contains(house_number, case=False, na=False)  # House number (9th column)
            if road:
                condition &= df.iloc[:, 10].str.contains(road, case=False, na=False)  # Road (11th column)
            if flat_number:
                condition &= df.iloc[:, 5].astype(str).str.contains(flat_number, case=False, na=False)  # Flat number (6th column)

            filtered_data = df[condition]

            if not filtered_data.empty:
                return filtered_data.iloc[0, 2], filtered_data.iloc[0, 3]  # Price (3rd column) & Date (4th column)

        except FileNotFoundError:
            st.error(f"{file} not found!")
        except Exception as e:
            st.error(f"Error reading {file}: {e}")

    return None, None

def predict_price_growth(region, house_type, historic_price, historic_year):
    """Predicts the price growth from the historic year to the present day using the growth rates."""
    try:
        # Load the growth data
        growth_data = pd.read_csv("house_price_growth.csv")

        # Filter growth data for the selected region and house type
        filtered_growth = growth_data[
            (growth_data["Region"] == region) & 
            (growth_data["House_Type"] == house_type) & 
            (growth_data["Year"] >= historic_year) & 
            (growth_data["Year"] <= 2024)  # Stop at 2024
        ]

        if filtered_growth.empty:
            st.warning("No growth data available for the selected region and house type.")
            return None

        # Calculate predicted prices with compounding growth
        predicted_prices = []
        years = []
        current_price = historic_price

        for _, row in filtered_growth.sort_values(by="Year").iterrows():
            growth_rate = row["Growth"] / 100  # Convert percentage to decimal
            current_price *= (1 + growth_rate)  # Compound the growth
            predicted_prices.append(current_price)
            years.append(row["Year"])

        # Create a DataFrame for the predicted prices
        predicted_df = pd.DataFrame({
            "Year": years,
            "Predicted Price": predicted_prices,
            "Growth (%)": filtered_growth.sort_values(by="Year")["Growth"].values
        })

        return predicted_df

    except Exception as e:
        st.error(f"Error predicting price growth: {e}")
        return None

def show():
    st.title("House Price Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("House Number")
        house_number = st.text_input("(optional)", key="house_number", max_chars=10)
    with col2:
        st.subheader("Road")
        road = st.text_input("(optional)", key="road", max_chars=30)
    with col3:
        st.subheader("Postcode")
        postcode = st.text_input("(Mandatory)", key="postcode", max_chars=10)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Bedrooms")
        bedroom_toggle = st.selectbox("Select the number of bedrooms", ["Unknown"] + [str(i) for i in range(1, 16)], index=0)
    with col2:
        st.subheader("House Type")
        house_type = st.selectbox("Select the house type", ["Unknown", "Semi-Detached", "Detached", "Terraced", "Flat"], index=0)
    with col3:
        st.subheader("Borough")
        boroughs = [
            "Barking And Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "Croydon", "Ealing", "Enfield", 
            "Greenwich", "Hackney", "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington And Chelsea", 
            "Kingston Upon Thames", "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", "Richmond Upon Thames", 
            "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
        ]
        boroughs.sort()
        selected_borough = st.selectbox("Select a Borough", ["Unknown"] + boroughs, index=0)

    # Add flat number input if house type is "Flat"
    flat_number = None
    if house_type == "Flat":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("")  # Empty column for spacing
        with col2:
            flat_number = st.text_input("Flat Number (optional)", key="flat_number", max_chars=10)
        with col3:
            st.write("")  # Empty column for spacing

    search_button = st.button("Search", key="search_button", use_container_width=True)
    clear_button = st.button("Clear", key="clear_button", use_container_width=True)

    if clear_button:
        st.session_state.clear()

    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None
    if "map_location" not in st.session_state:
        st.session_state.map_location = None
    if "historic_price" not in st.session_state:
        st.session_state.historic_price = None
    if "historic_date" not in st.session_state:
        st.session_state.historic_date = None

    def update_prediction():
        if not postcode:
            st.error("Please enter a postcode.")
            return
        if not is_valid_uk_postcode(postcode):
            st.error("Please enter a valid UK postcode.")
            return

        house_price, date = search_house_details(postcode, house_number, road, flat_number)

        if house_price and date:
            st.session_state.historic_price = house_price
            st.session_state.historic_date = pd.to_datetime(date).strftime('%d %B %Y')
            lat, lon = get_coordinates(postcode)
            st.session_state.map_location = (lat, lon) if lat and lon else None
        else:
            st.session_state.predicted_price = "No matching house found. Please check your inputs."
            st.session_state.map_location = None

    if search_button:
        update_prediction()

    if st.session_state.map_location:
        st.subheader("Postcode Location on Map")

        # Create a layout with two columns for the map and historic price box
        col1, col2 = st.columns([2, 1])

        # Display the map on the left
        with col1:
            lat, lon = st.session_state.map_location
            map_ = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker([lat, lon], popup="Property Location", tooltip="Click for details", icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(map_)
            st_folium(map_, width=500, height=400)

        # Display the historic selling price in a box on the right
        with col2:
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                f"<h3>Historic Selling Price</h3>"
                f"<p style='font-size: 24px; font-weight: bold;'>£{st.session_state.historic_price:,.2f}</p>"
                f"<p style='font-size: 16px;'>{st.session_state.historic_date}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
    elif search_button:
        st.warning("No valid location found for the given postcode.")

    st.subheader("Historic Selling Prices")
    if st.session_state.historic_price and st.session_state.historic_date:
        # Predict price growth
        if house_type != "Unknown" and selected_borough != "Unknown":
            # Map house type to the corresponding code (D, F, O, S, T)
            house_type_map = {
                "Detached": "D",
                "Flat": "F",
                "Other": "O",
                "Semi-Detached": "S",
                "Terraced": "T"
            }
            house_type_code = house_type_map.get(house_type, "")

            if house_type_code:
                historic_year = pd.to_datetime(st.session_state.historic_date).year
                predicted_df = predict_price_growth(selected_borough, house_type_code, st.session_state.historic_price, historic_year)

                if predicted_df is not None:
                    st.subheader("Predicted Price Growth")

                    # Create a layout with two columns
                    col1, col2 = st.columns([1, 2])

                    # Display the final predicted price in a box on the left
                    with col1:
                        final_price = predicted_df["Predicted Price"].iloc[-1]
                        st.markdown(
                            f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                            f"<h3>Predicted House Price</h3>"
                            f"<p style='font-size: 24px; font-weight: bold;'>£{final_price / 1e6:.2f}M</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    # Plot the predicted prices on the right
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(predicted_df["Year"], predicted_df["Predicted Price"] / 1e6, marker="o", linestyle="-", color="b")
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Price (£Million)")
                        ax.set_title(f"Predicted Price Growth for {selected_borough} ({house_type})")

                        # Format x-axis to show full years without decimals
                        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

                        # Add percentage increases as annotations with smaller font size
                        for i, row in predicted_df.iterrows():
                            ax.annotate(
                                f"{row['Growth (%)']:.1f}%",
                                (row["Year"], row["Predicted Price"] / 1e6),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha="center",
                                fontsize=8
                            )

                        st.pyplot(fig)
    else:
        st.write("No historic price available.")

if __name__ == "_main_":
    show()