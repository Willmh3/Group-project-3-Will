import streamlit as st
import re
import pandas as pd
import numpy as np

data = pd.read_parquet("FinalData.parquet")
pubData = pd.read_csv("pubData.csv")
wardScore = pd.read_csv("wardScore.csv")
pubList = pubData['pub_name_ward'].tolist()
wardToLsoa = pd.read_csv("LsoaToWard.csv")
DEBUG = False
maxPrice = data['price'].max()
minPrice = data['price'].min()

londonBoroughs = [
    "Show All", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", 
    "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", 
    "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", 
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames", 
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", 
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", 
    "Wandsworth", "Westminster"
]
column_mapping = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "price": "Property Price",
    "dateOfTransfer": "Transaction Date",
    "borough": "Borough",
    "houseType": "House Type",
    "numberOfBedrooms": "Bedrooms"
}

def pSearch(hLong, hLat, pLong, pLat):

    distance = ((hLong - pLong) ** 2 + (pLat - hLat) ** 2) ** 0.5

    distance *= 60

    if distance < 1:
        return True
    else:
        return False
    
def getWardScore(wardScores, lsoa):
    if st.session_state['greenSpaceCheck'] == False:
        G = 0
    else:
        G = st.session_state['greenSpace']

    if st.session_state['healthCheck'] == False:
        H = 0
    else:
        H = st.session_state['health']

    if st.session_state['educationCheck'] == False:
        E = 0
    else:
        E = st.session_state['education']

    if st.session_state['safetyCheck'] == False:
        S = 0
    else:
        S = st.session_state['safety']

    if st.session_state['transportCheck'] == False:
        T = 0
    else:
        T = st.session_state['transport']

    # Try to find the corresponding ward for the given LSOA
    ward_match = wardToLsoa.loc[wardToLsoa["LSOA21CD"] == lsoa, "Formatted_Ward"]

    if ward_match.empty:
        st.write(f"⚠ No matching ward found for LSOA: {lsoa}") if DEBUG else None
        return 0  # Default score if LSOA isn't found

    ward = ward_match.iloc[0]
    st.write(f"✅ Found ward: {ward} for LSOA: {lsoa}") if DEBUG else None

    columns = ["Green_Space_Score", "Health_Score", "Education_Score", "Safety_Score", "Transport_Score"]
    
    # Try to find scores for the ward
    scores = wardScores.loc[wardScores["Ward name"] == ward, columns]

    if scores.empty:
        st.write(f"⚠ No scores found for ward: {ward}") if DEBUG else None
        return 0  # Default score if no data exists for the ward

    scores = scores.values.flatten().tolist()
    st.write(f"🏆 Scores for {ward}: {scores}") if DEBUG else None

    totScore = scores[0] * G + scores[1] * H + scores[2] * E + scores[3] * S + scores[4] * T
    st.write(f"🎯 Computed Score for {ward}: {totScore}") if DEBUG else None

    return totScore

    



def show():
    maxPrice = data['price'].max()
    minPrice = data['price'].min()
    st.title("🔍 House Searching")
    st.write("This section will help you search for houses based on different criteria.")
    st.write(data.head()) if DEBUG else None
    st.write(data["house_Type"].unique()) if DEBUG else None
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

    st.write(getWardScore(wardScore, "E01000032")) if DEBUG else None
    col1, col2 = st.columns([1, 2])
    with col1:
        
        st.markdown("### Search Criteria")
        postcode = st.text_input("Postcode:")
        borough = st.selectbox("Borough:", londonBoroughs)
        houseType = st.selectbox("House Type:", ["Show All", "F", "T", "S", "O", "D"])
        numBedrooms = st.slider("Number of Bedrooms:", min_value=1, max_value=10, value=1, step=1)
        maxPrice = st.slider("Maximum Price:", min_value=int(minPrice), max_value=int(maxPrice), value=1000000, step=50000)
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("Advanced"):
                st.session_state['advancedShow'] = not st.session_state['advancedShow']

        with col1b:
            if st.button("Search"):
                st.session_state['advancedShow'] = False
                filteredData = data.copy()
                st.write(f"Before postcode filter: {len(filteredData)} rows") if DEBUG else None

                if postcode:
                    filteredData = filteredData[filteredData["postcode"].str.contains(postcode, na=False, case=False)]
                st.write(f"After postcode filter: {len(filteredData)} rows") if DEBUG else None

                if numBedrooms:
                    filteredData = filteredData[filteredData["numberOfBedrooms"] == numBedrooms]
                st.write(f"After bedroom filter: {len(filteredData)} rows") if DEBUG else None

                if maxPrice:
                    filteredData = filteredData[filteredData["price"] <= maxPrice]
                st.write(f"After price filter: {len(filteredData)} rows") if DEBUG else None

                if houseType != "Show All":
                    filteredData = filteredData[filteredData["house_Type"] == houseType]
                st.write(filteredData["house_Type"].unique()) if DEBUG else None
                st.write(f"After house type filter: {len(filteredData)} rows") if DEBUG else None

                if borough != "Show All":
                    filteredData = filteredData[filteredData["borough"] == borough.upper()]
                st.write(f"After borough filter: {len(filteredData)} rows") if DEBUG else None

                if st.session_state['beerCheck']:
                    pubLong = pubData.loc[pubData['pub_name_ward'] == st.session_state['beer'], 'longitude']
                    pubLat = pubData.loc[pubData['pub_name_ward'] == st.session_state['beer'], 'latitude']
                    filteredData = filteredData[
                        filteredData.apply(
                            lambda row: pSearch(row["longitude"], row["latitude"], pubLong.values[0], pubLat.values[0]),
                            axis=1
                        )
                    ]
                st.write(f"After beer filter: {len(filteredData)} rows") if DEBUG else None

                filteredData = filteredData.sort_values(by="price", ascending=False).head(10)
                st.session_state['filteredData'] = filteredData

    with col2:
        if not st.session_state['advancedShow']:
            st.markdown("### Results Table")
        if st.session_state['advancedShow']:
            st.markdown("### Advanced Search")
            st.write("Select what matters to you xx")
            colx, coly, colz = st.columns([1, 1, 1])
            with colx:
                st.session_state['greenSpaceCheck'] = st.checkbox("Green Space", value = st.session_state['greenSpaceCheck'])
                st.session_state['safetyCheck'] = st.checkbox("Safety", value = st.session_state['safetyCheck'])
            with coly:
                st.session_state['healthCheck'] = st.checkbox("Health", value = st.session_state['healthCheck'])
                st.session_state['transportCheck'] = st.checkbox("Transport", value = st.session_state['transportCheck'])
            with colz:
                st.session_state['educationCheck'] = st.checkbox("Education", value = st.session_state['educationCheck'])
                st.session_state['beerCheck'] = st.checkbox("Beer", value = st.session_state['beerCheck'])

            col2a, col2b = st.columns([1,1])

            
            with col2a:
                if st.session_state['greenSpaceCheck']:
                    st.session_state['greenSpace'] = st.slider("Green Space:", min_value=1, max_value=10, 
                        step=1, key="greenSpaceSlider", value = st.session_state['greenSpace'])

                
                if st.session_state['healthCheck']:
                    st.session_state['health'] = st.slider("Health:", min_value=1, max_value=10, 
                        step=1, key="healthSlider", value = st.session_state['health'])
                
                if st.session_state['educationCheck']:
                    st.session_state['education'] = st.slider("Education:", min_value=1, max_value=10, 
                        step=1, key="educationSlider", value = st.session_state['education'])
            
            with col2b:
                if st.session_state['safetyCheck']:
                    st.session_state['safety'] = st.slider("Safety:", min_value=1, max_value=10, 
                        step=1, key="safetySlider", value = st.session_state['safety'])
                    
                if st.session_state['transportCheck']:
                    st.session_state['transport'] = st.slider("Transport:", min_value=1, max_value=10, 
                        step=1, key="transportSlider", value = st.session_state['transport'])
                    
                if st.session_state['beerCheck']:
                    st.session_state['beer'] = st.selectbox("Pub:", pubList)



        elif not st.session_state['filteredData'].empty:
            if "lsoaCode" in st.session_state['filteredData'].columns:
                st.session_state['filteredData']["Ward Score"] = st.session_state['filteredData']["lsoaCode"].apply(
                    lambda lsoa: getWardScore(wardScore, lsoa) if pd.notna(lsoa) else None
                )
            else:
                st.write("big error") if DEBUG else None
            displayType = st.radio("Display Type:", ["Table", "Map"], horizontal=True, help="Choose how to display your results")

            if displayType == "Table":
                displayData = st.session_state['filteredData'].rename(columns=column_mapping)
                st.dataframe(displayData, height=410)
            else:
                mapData = st.session_state['filteredData'].dropna(subset=["latitude", "longitude"])
                if not mapData.empty:
                    st.map(mapData, height=410)
                else:
                    st.write("No valid location data to display on the map.")

        elif not st.session_state['filteredData'].empty:
            st.write("No matching results found.")

        else:
            st.write("Please enter search criteria.")