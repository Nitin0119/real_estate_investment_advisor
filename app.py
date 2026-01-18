import streamlit as st
import pandas as pd
import joblib

from src.preprocessing import preprocess_base
from src.feature_engineering import add_engineered_features

CITY_TIER_MAP = {
    "Mumbai": "Tier_1",
    "Delhi": "Tier_1",
    "Bangalore": "Tier_1",
    "Chennai": "Tier_1",
    "Hyderabad": "Tier_1",
    "Pune": "Tier_2",
    "Noida": "Tier_2",
    "Gurgaon": "Tier_2",
    "Faridabad": "Tier_2",
    "Ghaziabad": "Tier_2"
}

def get_city_tier(city):
    return CITY_TIER_MAP.get(city, "Tier_3")

@st.cache_resource
def load_models():
    clf = joblib.load('models/investment_classifier.pkl')
    reg = joblib.load('models/investment_regressor.pkl')
    return clf, reg

classifier, regressor = load_models()

"""
Streamlit Page Setup
"""
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="centered"
)

st.title("🏠 Real Estate Investment Advisor")
st.write("Evaluate whether a property is a good investment and predict its value after 5 years.")

"""
User Input Form
"""

with st.form(key='property_input_form'):
    st.subheader("Property Details")
    #later add 42 city albhabet wise
    city = st.selectbox("city", sorted([
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"
    ]))

    property_type = st.selectbox(
        "Property Type", ["Apartment", "Independent House", "Villa"]
    )

    bhk = st.number_input("BHK", min_value=1, max_value=5, step=1, value=2)
    size = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1000)

    price = st.number_input(
        "Current Price (Lakhs)",
        min_value=10.0,
        max_value=1000.0,
        value=100.0
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1950,
        max_value=2025,
        value=2015
    )

    transport = st.selectbox(
        "Public Transport Accessibility",
        ['Low', 'Medium', 'High']
    )

    schools = st.slider("Nearby Schools", 0, 10, 3)
    hospitals = st.slider("Hospitals", 0, 10, 2)

    amenities = st.multiselect(
        "Amenities",
        ['Playground', 'Gym', 'Garden', 'Pool', 'Clubhouse', 'Lift']
    )

    parking = st.selectbox("Parking Space", ["Yes", "No"])
    security = st.selectbox("Security", ["Yes", "No"])
    furnished = st.selectbox("Furnishing Status", ['Furnished', 'Semi_furnished', 'Unfurnished'])
    availability = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])

    submit = st.form_submit_button("Evaluate Investment")

    city_tier = get_city_tier(city)
    if submit:
        input_df = pd.DataFrame([{
            "city": city,
            "city_tier": city_tier,
            "property_type": property_type,
            "bhk": bhk,
            "size_in_sqft": size,
            "price_in_lakhs": price,
            "year_built": year_built,
            "public_transport_accessibility": transport,
            "nearby_schools": schools,
            "nearby_hospitals": hospitals,
            "amenities": amenities,
            "parking_space": parking,
            "security": security,
            "furnished_status": furnished,
            "availability_status": availability,
            "owner_type": owner_type,
            "state":"NA"
        }])

        processed_df = preprocess_base(input_df)
        processed_df = add_engineered_features(processed_df)

        """
        Make Predictions
        """

        investment_pred = classifier.predict(processed_df)[0]
        investment_prob = classifier.predict_proba(processed_df)[0][1]

        future_price = regressor.predict(processed_df)[0]

        """
        Display Results
        """

        st.subheader("Investment Results")

        if investment_pred == 1:
            st.success(f"Good Investment (confidence: {investment_prob * 100:.2f}%)")
        else:
            st.error(f"Not a Good Investment (confidence: {(1-investment_prob) * 100:.2f}%)")
        st.metric(
            label="Estimated Property Price After 5 Years",
            value=f"₹ {future_price:,.2f} Lakhs"
        )