import streamlit as st

def show():
    st.title("Data Analytics")

    # Dropdown for selecting analytics option
    analytics_option = st.selectbox(
        "Choose an option for data analytics:",
        ["Crime Rate", "House Growth", "Average Price"]
    )

    # Display content based on the selected option below
    if analytics_option == "Crime Rate":
        st.subheader("Crime Rate Analysis")
        st.write("Display crime rate analysis here.")
        # Add your code for crime rate analysis (e.g., display graphs, statistics, etc.)

    elif analytics_option == "House Growth":
        st.subheader("House Growth Analysis")
        st.write("Display house growth analysis here.")
        # Add your code for house growth analysis (e.g., display house price growth charts)

    elif analytics_option == "Average Price":
        st.subheader("Average Price Analysis")
        st.write("Display average price analysis here.")
        # Add your code for average price analysis (e.g., show price comparison graphs, stats)