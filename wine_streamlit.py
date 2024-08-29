import streamlit as st
import numpy as np

from Used_Models import prediction

# Set the page configuration
st.set_page_config(page_title="RED WINE QUALITY PREDICTION", page_icon=":material/model_training:", layout="wide")

# columns
left_col, main_col = st.columns([4, 10])

# For user input
x_test = []

# Left for wine variables
with left_col:
    st.header("Wine Variables")
    
    # Wrap the inputs in a scrollable container
    st.markdown("""
        <div style="height: 80%; overflow-y: auto; padding-right: 10px;">
    """, unsafe_allow_html=True)

    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=1.0, step=0.1)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, step=0.1)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.000, max_value=1.000, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=0.1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=0.1)
    density = st.number_input("Density", min_value=0.0, max_value=1.0, step=0.1, format="%.4f")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    sulphates = st.number_input("Sulphates", min_value=0.0, step=0.1, format="%.3f")
    alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)
    
    x_test.extend([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol])
    st.markdown("</div>", unsafe_allow_html=True) 

# Right for prediction
with main_col:
    st.header("RED WINE QUALITY PREDICTION")
    
    if st.button("Predict", key="predict", on_click=None):
        # prediction = random.choice([5, 6, 7, 4, 8, 3])
        prediction_quality = prediction(np.array(x_test).reshape(1, -1))
        st.subheader(f"The predicted Wine quality is: {prediction_quality}")

    # To add space 
    st.write("")
    st.markdown("## How it works")
    st.markdown("""
 Using the Portuguese Verde model, this website predicts wine quality based on user-entered features. Simply input the relevant information on the left to receive your wine's quality rating from 3 to 8.
    """)
 # Add a white line
    st.markdown("""
        <hr style="border:1px solid white; width: 100%; margin-top: 40px;">
    """, unsafe_allow_html=True)

    # Add logos and release by
    st.markdown("""
        <div style="text-align: end; font-size: 16px; color: white;">
            Released by : 
            <a href="https://www.linkedin.com/in/omer-alt-fotso-992624244/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20px"></a>
            <a href="https://www.linkedin.com/in/miorar/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20px"></a>
            <a href="https://www.linkedin.com/in/profile3/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20px"></a>
            <a href="https://github.com/yourgithub/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20px"></a>
        </div>
    """, unsafe_allow_html=True)

# Customize the text size and other styles
st.markdown("""
    <style>
    /* Set text size for all text elements */
    .css-1v0mbdj p, .css-1v0mbdj h2, .css-1v0mbdj h1, .css-1v0mbdj h3, .css-1v0mbdj li {
        font-size: 16px !important;
    }

    .stButton button {
        color: white;
    }
    .stButton button:hover {
        background-color: #6cd625;
    }
    /*.css-18e3th9 {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }*/
    .block-container {
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# CSS for the Predict button
# st.markdown("""
#     <style>
#     .stButton button {
#         background-color: #6cd625;
#         color: white;
#     }
#     .stButton button:hover {
#         background-color: #6cd625;
#     }
#     </style>
# """, unsafe_allow_html=True)
