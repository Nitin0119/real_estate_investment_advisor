RANDOM_STATE = 42

TARGET_CLASSIFICATION = "good_investment"
TARGET_REGRESSION = "future_price_5y"

ID_COL = "id"

NUMERIC_FEATURES = [
    "bhk",
    "size_in_sqft",
    "price_in_lakhs",
    "year_built",
    "nearby_schools",
    "nearby_hospitals"
]

ORDINAL_FEATURES = {
    "public_transport_accessibility": ["Low", "Medium", "High"]
}

CATEGORICAL_FEATURES = [
    "state",
    "city",
    "locality",
    "property_type",
    "furnished_status",
    "parking_space",
    "security",
    "amenities",
    "facing",
    "owner_type",
    "availability_status"
]

DROP_COLUMNS = [
    'age_of_property',
    'floor_no',
    'id',
    'price_per_sqft',
    'total_floors'
]
