import streamlit as st

def show():
    st.title("Credits")
    st.write("Meet the team of mathematicians behind this operation:")
    
    st.subheader("Team Members")
    st.write("""
    - **John Doe**: Data Scientist
    - **Jane Smith**: Machine Learning Engineer
    - **Alice Johnson**: Frontend Developer
    - **Bob Brown**: Backend Developer
    """)
    
    st.subheader("Special Thanks")
    st.write("We would like to thank our mentors and collaborators for their support.")