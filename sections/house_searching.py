import streamlit as st
import re
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# Load data with error handling
try:
    data = pd.read_csv("dummyData.csv")
    pubData = pd.read_csv("pubData.csv")
except FileNotFoundError as e:
    st.error(f"Error loading data file: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

pubList = pubData['pub_name_ward'].tolist()

londonBoroughs = [
    "Show All", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", 
    "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", 
    "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", 
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames", 
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", 
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", 
    "Wandsworth", "Westminster"
]

def pSearch(hLong, hLat, pLong, pLat):
    """Check if a house is within 1 mile of a pub."""
    distance = np.sqrt((hLong - pLong) ** 2 + (hLat - pLat) ** 2) * 60
    return distance < 1

def is_valid_uk_postcode(postcode):
    """Validates a UK postcode using a regular expression."""
    pattern = r"^([A-Za-z]{1,2}\d{1,2}[A-Za-z]? \d[A-Za-z]{2})$"
    return re.match(pattern, postcode) is not None

def show():
    st.title("🔍 House Searching")
    st.write("This section will help you search for houses based on different criteria.")

    # Initialize session state variables
    if 'filteredData' not in st.session_state:
        st.session_state['filteredData'] = pd.DataFrame()
    if 'advancedShow' not in st.session_state:
        st.session_state['advancedShow'] = False
    if 'greenSpaceCheck' not in st.session_state:
        st.session_state['greenSpaceCheck'] = False
    if 'greenSpace' not in st.session_state:
        st.session_state['greenSpace'] = 5
    if 'healthCheck' not in st.session_state:
        st.session_state['healthCheck'] = False
    if 'health' not in st.session_state:
        st.session_state['health'] = 5
    if 'educationCheck' not in st.session_state:
        st.session_state['educationCheck'] = False
    if 'education' not in st.session_state:
        st.session_state['education'] = 5
    if 'safetyCheck' not in st.session_state:
        st.session_state['safetyCheck'] = False
    if 'safety' not in st.session_state:
        st.session_state['safety'] = 5
    if 'transportCheck' not in st.session_state:
        st.session_state['transportCheck'] = False
    if 'transport' not in st.session_state:
        st.session_state['transport'] = 5
    if 'beerCheck' not in st.session_state:
        st.session_state['beerCheck'] = False
    if 'beer' not in st.session_state:
        st.session_state['beer'] = 5

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Search Criteria")
        postcode = st.text_input("Outward Postcode:")
        if postcode and not is_valid_uk_postcode(postcode):
            st.error("Please enter a valid UK postcode.")
            st.stop()

        borough = st.selectbox("Borough:", londonBoroughs)
        houseType = st.selectbox("House Type:", ["Show All", "All", "Flat", "Detached", "Semi Detached", "Not Detached"])
        numBedrooms = st.slider("Number of Bedrooms:", min_value=1, max_value=10, value=1, step=1)
        maxPrice = st.slider("Maximum Price:", min_value=50000, max_value=2000000, value=500000, step=50000)

        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("Advanced"):
                st.session_state['advancedShow'] = not st.session_state['advancedShow']
        with col1b:
            if st.button("Reset Filters"):
                st.session_state['filteredData'] = pd.DataFrame()
                st.session_state['advancedShow'] = False
                st.session_state['greenSpaceCheck'] = False
                st.session_state['greenSpace'] = 5
                st.session_state['healthCheck'] = False
                st.session_state['health'] = 5
                st.session_state['educationCheck'] = False
                st.session_state['education'] = 5
                st.session_state['safetyCheck'] = False
                st.session_state['safety'] = 5
                st.session_state['transportCheck'] = False
                st.session_state['transport'] = 5
                st.session_state['beerCheck'] = False
                st.session_state['beer'] = 5
                st.rerun()

        if st.button("Search"):
            st.session_state['advancedShow'] = False
            filteredData = data.copy()
            if postcode:
                filteredData = filteredData[filteredData["postcode"].str.contains(postcode, na=False, case=False)]
            if numBedrooms:
                filteredData = filteredData[filteredData["numberOfBedrooms"] == numBedrooms]
            if maxPrice:
                filteredData = filteredData[filteredData["price"] <= maxPrice]
            if houseType != "Show All":
                filteredData = filteredData[filteredData["houseType"] == houseType]
            if borough != "Show All":
                filteredData = filteredData[filteredData["borough"] == borough]
            if st.session_state['beerCheck']:
                pubLong = pubData.loc[pubData['pub_name_ward'] == st.session_state['beer'], 'longitude']
                pubLat = pubData.loc[pubData['pub_name_ward'] == st.session_state['beer'], 'latitude']
                filteredData = filteredData[
                    filteredData.apply(
                        lambda row: pSearch(row["longitude"], row["latitude"], pubLong.values[0], pubLat.values[0]),
                        axis=1
                    )
                ]

            filteredData = filteredData.sort_values(by="price", ascending=False).head(10)
            st.session_state['filteredData'] = filteredData

    with col2:
        if not st.session_state['advancedShow']:
            st.markdown("### Results Table")
        if st.session_state['advancedShow']:
            st.markdown("### Advanced Search")
            st.write("Select what matters to you:")
            colx, coly, colz = st.columns([1, 1, 1])
            with colx:
                st.session_state['greenSpaceCheck'] = st.checkbox("Green Space", value=st.session_state['greenSpaceCheck'])
                st.session_state['safetyCheck'] = st.checkbox("Safety", value=st.session_state['safetyCheck'])
            with coly:
                st.session_state['healthCheck'] = st.checkbox("Health", value=st.session_state['healthCheck'])
                st.session_state['transportCheck'] = st.checkbox("Transport", value=st.session_state['transportCheck'])
            with colz:
                st.session_state['educationCheck'] = st.checkbox("Education", value=st.session_state['educationCheck'])
                st.session_state['beerCheck'] = st.checkbox("Beer", value=st.session_state['beerCheck'])

            col2a, col2b = st.columns([1, 1])
            with col2a:
                if st.session_state['greenSpaceCheck']:
                    st.session_state['greenSpace'] = st.slider(
                        "Green Space:", min_value=1, max_value=10, step=1, 
                        key="greenSpaceSlider", value=st.session_state['greenSpace'],
                        help="Higher values indicate more green space in the area."
                    )
                if st.session_state['healthCheck']:
                    st.session_state['health'] = st.slider(
                        "Health:", min_value=1, max_value=10, step=1, 
                        key="healthSlider", value=st.session_state['health'],
                        help="Higher values indicate better healthcare access."
                    )
                if st.session_state['educationCheck']:
                    st.session_state['education'] = st.slider(
                        "Education:", min_value=1, max_value=10, step=1, 
                        key="educationSlider", value=st.session_state['education'],
                        help="Higher values indicate better educational facilities."
                    )
            with col2b:
                if st.session_state['safetyCheck']:
                    st.session_state['safety'] = st.slider(
                        "Safety:", min_value=1, max_value=10, step=1, 
                        key="safetySlider", value=st.session_state['safety'],
                        help="Higher values indicate safer areas."
                    )
                if st.session_state['transportCheck']:
                    st.session_state['transport'] = st.slider(
                        "Transport:", min_value=1, max_value=10, step=1, 
                        key="transportSlider", value=st.session_state['transport'],
                        help="Higher values indicate better transport links."
                    )
                if st.session_state['beerCheck']:
                    st.session_state['beer'] = st.selectbox("Pub:", pubList)

        elif not st.session_state['filteredData'].empty:
            displayType = st.radio("Display Type:", ["Table", "Map"], horizontal=True, help="Choose how to display your results")

            if displayType == "Table":
                st.dataframe(st.session_state['filteredData'], height=410)
            else:
                mapData = st.session_state['filteredData'].dropna(subset=["latitude", "longitude"])
                if not mapData.empty:
                    m = folium.Map(location=[mapData["latitude"].mean(), mapData["longitude"].mean()], zoom_start=12)
                    for _, row in mapData.iterrows():
                        folium.Marker(
                            [row["latitude"], row["longitude"]],
                            popup=f"Price: £{row['price']:,}<br>Bedrooms: {row['numberOfBedrooms']}",
                            tooltip="Click for details",
                            icon=folium.Icon(color="red", icon="home", prefix="fa")
                        ).add_to(m)
                    st_folium(m, width=700, height=410)
                else:
                    st.write("No valid location data to display on the map.")

        elif not st.session_state['filteredData'].empty:
            st.write("No matching results found.")

        else:
            st.write("Please enter search criteria.")

if __name__ == "_main_":
    show()