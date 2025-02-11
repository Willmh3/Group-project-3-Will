import streamlit as st
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def show():
    st.title("House Price Prediction")

    # Input fields
    house_number = st.text_input("House Number (optional)", key="house_number")
    road = st.text_input("Road (optional)", key="road")
    postcode = st.text_input("Postcode (optional)", key="postcode")
    flat_number = st.text_input("Flat Number (optional)", key="flat_number")

    # Initialize session state variables if they don't exist
    if "address" not in st.session_state:
        st.session_state.address = ""
    if "latitude" not in st.session_state:
        st.session_state.latitude = None
    if "longitude" not in st.session_state:
        st.session_state.longitude = None

    # Function to get location
    def get_location(address):
        geolocator = Nominatim(user_agent="geoapiExercises")
        try:
            location = geolocator.geocode(address)
            return (location.latitude, location.longitude) if location else (None, None)
        except GeocoderTimedOut:
            st.error("Geocoder service timed out. Please try again.")
            return None, None

    # Show the result of the geolocation (if available)
    if st.button("Get Location"):
        address = f"{house_number} {road} {flat_number} {postcode}".strip()

        if not address:
            st.error("Please enter a valid address.")
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
                else:
                    st.error("Address not found. Please check the details.")

    # Show the location map if the latitude and longitude are available
    if st.session_state.latitude and st.session_state.longitude:
        st.write(f"You entered: {st.session_state.address}")

        # Create a map using the latitude and longitude
        map_ = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=15)
        folium.Marker([st.session_state.latitude, st.session_state.longitude], popup=st.session_state.address).add_to(map_)

        # Display the map
        st.write("Map showing the location:")
        st.components.v1.html(map_.repr_html(), width=700, height=500)