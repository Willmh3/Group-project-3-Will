import streamlit as st
import pandas as pd
import plotly.express as px
import json
import geopandas as gpd

DEBUG = False

@st.cache_resource
def loadGeojson():
    with open("londonBoroughs.geojson") as response:
        geojsonData = json.load(response)
    
    # Simplify geometry for faster loading
    gdf = gpd.GeoDataFrame.from_features(geojsonData["features"])
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.001, preserve_topology=True)
    
    return json.loads(gdf.to_json())

boroughScores = pd.read_csv("wardScore.csv")
geo = loadGeojson()

boroughs = [feature["properties"]["name"].strip() for feature in geo["features"]]

def plotChoropleth(selectedBorough, category):
    categoryMap = {
        "Green space": "Green_Space_Score",
        "Health": "Health_Score",
        "Education": "Education_Score",
        "Safety": "Safety_Score",
        "Transport": "Transport_Score",
        "Price per Square Foot": "Housing_Affordability_Score"
    }
    
    if category == "None":
        df = pd.DataFrame({"borough": boroughs, "score": [None] * len(boroughs)})
        colorScale = "gray"
    else:
        categoryColumn = categoryMap[category]
        df = boroughScores.groupby("Ward name")[categoryColumn].mean().reset_index()
        df.rename(columns={"Ward name": "borough", categoryColumn: "score"}, inplace=True)
        colorScale = "YlGnBu"

    fig = px.choropleth_mapbox(
        df,
        geojson=geo,
        locations="borough",
        featureidkey="properties.name",
        color="score",
        color_continuous_scale=colorScale,
        mapbox_style="carto-positron",
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=8.5,
    )
    
    fig.update_traces(marker_line_width=1, marker_line_color="black", hoverinfo="location")
    
    fig.update_layout(
        margin=dict(r=0, t=0, l=0, b=0),
        height=400,
        width=400,
        showlegend=True
    )
    
    return fig

def plotBoxPlot(selectedBorough, category):
    categoryMap = {
        "Green space": "Green_Space_Score",
        "Health": "Health_Score",
        "Education": "Education_Score",
        "Safety": "Safety_Score",
        "Transport": "Transport_Score",
        "Price per Square Foot": "Housing_Affordability_Score"
    }

    if category == "None":
        return None 
    
    categoryColumn = categoryMap[category]

    wardScoresInBorough = boroughScores[boroughScores["Ward name"].str.contains(selectedBorough + " ")]

    if wardScoresInBorough.empty:
        return None
    
    fig = px.box(
        wardScoresInBorough,
        x=categoryColumn,
        orientation="h",
        labels={categoryColumn: category},
        color_discrete_sequence=["#4CAF50"], 
    )
    fig.update_traces(hoverinfo="none")
    fig.update_layout(
        height=200,
        width=600,
        showlegend=False,
        margin=dict(r=0, t=0, l=0, b=0),
    )

    return fig

def show():
    st.title("🔍 Borough Analysis")
    cola, colb = st.columns([4, 2])  # Adjust column proportions as needed
    
    # Category selection comes here
    with cola:
        category = st.radio(
            "Select the category to analyse",
            ("None", "Green space", "Health", "Education", "Safety", "Transport", "Price per Square Foot"),
            horizontal=True
        )

    with colb:
        selectedBorough = st.selectbox("Select a Borough:", ["None"] + boroughs, index=0, key="borough_select")
        
        # Display the borough details if it's selected
        if selectedBorough != "None" and DEBUG:
            st.write(boroughScores.loc[boroughScores['Ward name'] == selectedBorough])

        # Plot the box plot in colb
        box_fig = plotBoxPlot(selectedBorough, category)
        if box_fig:
            st.plotly_chart(box_fig, use_container_width=True)
    
    # Plot the choropleth map in cola
    with cola:
        choropleth_fig = plotChoropleth(selectedBorough, category)
        st.plotly_chart(choropleth_fig, use_container_width=True)