import streamlit as st
import sections.home as home
import sections.house_price as house_price
import sections.data_analytics as data_analytics
import sections.house_searching as house_searching

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Maroon top bar
st.markdown(
    """
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
        .stButton>button {
            width: 100%;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="top-bar">House Price Predictor</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Initialize session state
if 'selected' not in st.session_state:
    st.session_state.selected = 'home'

# Define button function
def styled_button(label, page):
    return st.sidebar.button(label, key=page, use_container_width=True)

# Navigation buttons
if styled_button("🏡 Home", "home"):
    st.session_state.selected = "home"
    st.rerun()

if styled_button("📈 House Price Predictor", "house_price"):
    st.session_state.selected = "house_price"
    st.rerun()

if styled_button("📊 Data Analytics", "data_analytics"):
    st.session_state.selected = "data_analytics"
    st.rerun()

if styled_button("🔍 House Searching", "house_searching"):
    st.session_state.selected = "house_searching"
    st.rerun()

# Show content based on selection
if st.session_state.selected == 'home':
    home.show()

elif st.session_state.selected == 'house_price':
    house_price.show()

elif st.session_state.selected == 'data_analytics':
    data_analytics.show()

elif st.session_state.selected == 'house_searching':
    house_searching.show()