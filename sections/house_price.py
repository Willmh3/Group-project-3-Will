import streamlit as st
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeopyError
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
    if "address" not in st.session_state:
        st.session_state.address = ""
    if "latitude" not in st.session_state:
        st.session_state.latitude = None
    if "longitude" not in st.session_state:
        st.session_state.longitude = None
    if "predicted_price" not in st.session_state:
        st.session_state.predicted_price = None

    # Function to get location with retry mechanism
    def get_location(address, retries=3):
        geolocator = Nominatim(user_agent="geoapiExercises")
        for attempt in range(retries):
            try:
                location = geolocator.geocode(address, timeout=10)
                if location:
                    return location.latitude, location.longitude
                else:
                    st.error("Address not found. Please check the details and try again.")
                    return None, None
            except GeocoderTimedOut:
                if attempt < retries - 1:
                    st.warning(f"Retrying geolocation... ({attempt + 1}/{retries})")
                else:
                    st.error("Geocoder service timed out. Please try again later.")
                    return None, None
            except GeopyError as e:
                st.error(f"Geolocation failed: {e}")
                return None, None

    # Function to validate London postcode
    def is_valid_london_postcode(postcode):
        london_postcode_regex = re.compile(r"^(([A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2})|GIR 0AA)$", re.IGNORECASE)
        return london_postcode_regex.match(postcode) is not None

    # Function to update predicted price based on inputs
    def update_predicted_price():
        if postcode and is_valid_london_postcode(postcode):
            # This is a placeholder. Replace this with a call to your predictive model.
            st.session_state.predicted_price = "$350,000"  # Random value for now
        else:
            st.session_state.predicted_price = "Invalid or missing postcode. Please enter a valid London postcode."

    # Automatically update predicted price when inputs change
    update_predicted_price()

    # Show the result of the geolocation (if available)
    if st.button("Get Location"):
        address = f"{house_number} {road} {flat_number} {postcode}".strip()

        if not postcode or not is_valid_london_postcode(postcode):
            st.error("Please enter a valid London postcode.")
        else:
            # Store address in session state to preserve it across reruns
            st.session_state.address = address

            # Check if latitude and longitude are already stored in session state
            if st.session_state.latitude is None or st.session_state.longitude is None:
                latitude, longitude = get_location(address)

                if latitude and longitude:
                    # Store latitude and longitude in session state
                    st.session_state.latitude = latitude
                    st.session_state.longitude = longitude

    # Show the location map if the latitude and longitude are available
    if st.session_state.latitude and st.session_state.longitude:
        st.write(f"You entered: {st.session_state.address}")

        # Create a map using the latitude and longitude
        map_ = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=15)
        folium.Marker([st.session_state.latitude, st.session_state.longitude], popup=st.session_state.address).add_to(map_)

        # Display the map
        st.write("Map showing the location:")
        st.components.v1.html(map_.repr_html(), width=700, height=500)

    # Placeholder for the predicted price in a styled box
    st.subheader("Predicted House Price")
    price_box = f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px;">
        {st.session_state.predicted_price}
    </div>
    """
    st.markdown(price_box, unsafe_allow_html=True)
