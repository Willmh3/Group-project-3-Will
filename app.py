import streamlit as st
from geopy.geocoders import Nominatim
import folium

st.title("House Price Predictor")

address = st.text_input("Enter an address:")

if address:
    st.write(f"You entered: {address}")
    
    # Use Geopy to get coordinates for the address
    geolocator = Nominatim(user_agent="house_price_predictor")
    location = geolocator.geocode(address)

    if location:
        # Create a map with the location
        map = folium.Map(location=[location.latitude, location.longitude], zoom_start=12)
        folium.Marker([location.latitude, location.longitude], popup=address, icon=folium.Icon(color='red')).add_to(map)

        # Display map in Streamlit
        st.map(map)
    else:
        st.write("Address not found.")
