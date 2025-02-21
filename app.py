import streamlit as st
import sections.home as home
import sections.house_price as house_price
import sections.data_analytics as data_analytics
import sections.house_searching as house_searching
import sections.credits as credits

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

# Initialize session state
if 'selected' not in st.session_state:
    st.session_state.selected = 'home'
if 'history' not in st.session_state:
    st.session_state.history = []  # Track navigation history

# Sidebar navigation
st.sidebar.title("Navigation")

# Navigation buttons
pages = {
    "🏡 Home": "home",
    "📈 House Price Predictor": "house_price",
    "📊 Data Analytics": "data_analytics",
    "🔍 House Searching": "house_searching",
    "Credits": "credits"
}

for label, page in pages.items():
    if st.sidebar.button(label, key=page):
        st.session_state.history.append(st.session_state.selected)  # Save current page to history
        st.session_state.selected = page

# Add a "Back" button to return to the previous page
if st.session_state.history:
    if st.sidebar.button("⬅️ Back"):
        st.session_state.selected = st.session_state.history.pop()  # Go back to the previous page

# Show content based on selection
if st.session_state.selected == 'home':
    home.show()
elif st.session_state.selected == 'house_price':
    house_price.show()
elif st.session_state.selected == 'data_analytics':
    data_analytics.show()
elif st.session_state.selected == 'house_searching':
    house_searching.show()
elif st.session_state.selected == 'credits':
    credits.show()
else:
    st.error("Invalid page selection. Please try again.")