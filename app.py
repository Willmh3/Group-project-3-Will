import streamlit as st
import sections.house_price as house_price
import sections.data_analytics as data_analytics

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Maroon top bar
st.markdown("""
    <style>
        .top-bar {
            background-color: maroon;
            color: white;
            padding: 15px;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-bar">House Price Predictor</div>', unsafe_allow_html=True)

# Sidebar buttons for navigation
st.sidebar.title("Navigation")

# Define buttons and use session_state to track which button was pressed
if 'selected' not in st.session_state:
    st.session_state.selected = 'home'  # Default selection

home_button = st.sidebar.button("🏠 Home")
house_price_button = st.sidebar.button("📈 House Price Predictor")
data_analytics_button = st.sidebar.button("📊 Data Analytics")

if home_button:
    st.session_state.selected = 'home'
elif house_price_button:
    st.session_state.selected = 'house_price'
elif data_analytics_button:
    st.session_state.selected = 'data_analytics'

# Show content based on the selection in session state
if st.session_state.selected == 'home':
    st.write("Welcome to the House Price Predictor App! Use the sidebar to navigate.")

elif st.session_state.selected == 'house_price':
    house_price.show()

elif st.session_state.selected == 'data_analytics':
    data_analytics.show()