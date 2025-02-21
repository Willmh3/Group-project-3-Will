import streamlit as st

def show():
    # Set custom CSS for the home page styling
    st.markdown(
        """
        <style>
            .hero-section {
                background-color: #E9F7EF;
                padding: 60px;
                text-align: center;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                margin-bottom: 40px;
            }
            .hero-section h1 {
                font-size: 3.5em;
                color: #2C3E50;
                font-weight: bold;
                letter-spacing: 2px;
            }
            .hero-section p {
                font-size: 1.3em;
                color: #34495E;
                line-height: 1.6;
            }
            .feature-section {
                display: flex;
                justify-content: space-around;
                margin-top: 40px;
            }
            .feature-box {
                background-color: #F4F6F6;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                padding: 30px;
                width: 45%;
                text-align: center;
                transition: transform 0.3s ease;
            }
            .feature-box:hover {
                transform: scale(1.05);
            }
            .feature-box h3 {
                color: #1ABC9C;
                font-size: 1.8em;
            }
            .feature-box p {
                color: #2C3E50;
                font-size: 1.1em;
                line-height: 1.5;
            }
            .cta-button {
                background-color: #27AE60;
                color: white;
                padding: 15px 30px;
                font-size: 1.2em;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 40px;
            }
            .cta-button:hover {
                background-color: #2ECC71;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Hero section (Welcome text)
    st.markdown(
        """
        <div class="hero-section">
            <h1>Welcome to the House Price Predictor</h1>
            <p>Use our tool to explore and predict house prices based on data analysis and machine learning models. Start by navigating through the sidebar!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature section (Highlighting different sections)
    st.markdown(
        """
        <div class="feature-section">
            <div class="feature-box">
                <h3>Predict House Prices</h3>
                <p>Predict house prices for various locations based on key factors like area, type, and economic growth.</p>
            </div>
            <div class="feature-box">
                <h3>Explore Data Analytics</h3>
                <p>Explore detailed analytics of housing market trends using interactive graphs and insights.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    

if __name__ == "_main_":
    show()