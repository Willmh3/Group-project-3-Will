import streamlit as st
from datetime import datetime, timedelta
from price_prediction import predict_house_price_hybrid

def show():
    st.title("House Price Prediction")

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Postcode")
        postcode = st.text_input("Enter Postcode (e.g., SW1A 1AA)", key="postcode", max_chars=10)
    with col2:
        st.subheader("Borough")
        boroughs = [
            "Barking And Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "Croydon", "Ealing", "Enfield", 
            "Greenwich", "Hackney", "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington And Chelsea", 
            "Kingston Upon Thames", "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge", "Richmond Upon Thames", 
            "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
        ]
        boroughs.sort()
        selected_borough = st.selectbox("Select a Borough", ["Unknown"] + boroughs, index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("House Type (Optional)")
        house_type = st.selectbox("Select the house type", ["Unknown", "Semi-Detached", "Detached", "Terraced", "Flat"], index=0)
    with col2:
        st.subheader("Bedrooms (Optional)")
        bedroom_toggle = st.selectbox("Select the number of bedrooms", ["Unknown"] + [str(i) for i in range(1, 16)], index=0)

    # Prediction button
    predict_button = st.button("Predict Price", key="predict_button", use_container_width=True)

    if predict_button:
        if not postcode:
            st.error("Please enter a postcode.")
        else:
            # Prepare inputs for the prediction function
            inputs = {
                "ds": datetime.now().strftime("%Y-%m-%d"),  # Current date
                "tfarea": 100.0,  # Default value (can be adjusted)
                "numberrooms": int(bedroom_toggle) if bedroom_toggle != "Unknown" else None,
                "CURRENT_ENERGY_EFFICIENCY": 80,  # Default value
                "POTENTIAL_ENERGY_EFFICIENCY": 90,  # Default value
                "Postcode": postcode,
                "region": selected_borough if selected_borough != "Unknown" else None,
                "tenure_type": "Freehold",  # Default to Freehold
                "house_type": house_type if house_type != "Unknown" else None,
                "alpha": 0.7  # Default weight for residual correction
            }

            # Predict current price
            current_prediction = predict_house_price_hybrid(**inputs)

            # Predict future price (15 years from now)
            future_date = (datetime.now() + timedelta(days=365 * 15)).strftime("%Y-%m-%d")
            inputs["ds"] = future_date
            future_prediction = predict_house_price_hybrid(**inputs)

            # Display current price prediction
            st.subheader("Current Price Prediction")
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                f"<h3>Predicted House Price</h3>"
                f"<p style='font-size: 24px; font-weight: bold;'>£{current_prediction['Predicted_Price']:,.2f}</p>"
                f"<p style='font-size: 16px;'>Price Range: £{current_prediction['Price_Range'][0]:,.2f} - £{current_prediction['Price_Range'][1]:,.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Display 15-year future estimate
            st.subheader("15-Year Future Estimate")
            st.markdown(
                f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f0f0; text-align: center;'>"
                f"<h3>Predicted House Price in 15 Years</h3>"
                f"<p style='font-size: 24px; font-weight: bold;'>£{future_prediction['Predicted_Price']:,.2f}</p>"
                f"<p style='font-size: 16px;'>Price Range: £{future_prediction['Price_Range'][0]:,.2f} - £{future_prediction['Price_Range'][1]:,.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    show()