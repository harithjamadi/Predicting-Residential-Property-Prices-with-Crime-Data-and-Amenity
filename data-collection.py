import time
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split

# Load data
transaction_data = pd.read_excel("data_source/Open Transaction Data.xlsx")
transaction_data.ffill(inplace=True)

# Setup headers for APIs
HEADERS = {"User-Agent": "DataCollection/1.0 (hrth@gmail.com)"}

# Function to create search query for Nominatim
def create_query(row):
    parts = [
        str(row['Road Name']).strip(),
        str(row['District']).strip(),
        "Malaysia"
    ]
    return ", ".join([part for part in parts if part and part.lower() != 'nan' and part.lower() != 'none'])

# Function to get geolocation (latitude, longitude) from Nominatim
def get_geolocation(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    return np.nan, np.nan

# Function to get POIs from Overpass API given latitude, longitude
def get_poi(lat, lon, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["amenity"];
      way(around:{radius},{lat},{lon})["amenity"];
      relation(around:{radius},{lat},{lon})["amenity"];
    );
    out center;
    """
    response = requests.post(overpass_url, data={'data': query})
    if response.status_code == 200:
        data = response.json()
        amenities_found = set()
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            amenity = tags.get('amenity')
            if amenity:
                amenities_found.add(amenity)
        return amenities_found
    return set()

# Amenities of interest
target_amenities = [
    "school", "kindergarten", "university", "hospital",
    "clinic", "supermarket", "place_of_worship", "bus_station", "marketplace"
]

# Remove districts with <2 records to ensure stratification works
district_counts = transaction_data['District'].value_counts()
valid_districts = district_counts[district_counts >= 2].index
transaction_data = transaction_data[transaction_data['District'].isin(valid_districts)]

# Stratified random sample of 1000 by District
_, sample_data = train_test_split(
    transaction_data,
    test_size=1000,
    stratify=transaction_data['District'],
    random_state=42
)
sample_data = sample_data.reset_index(drop=True)

# Initialize query and POI columns
sample_data['query'] = sample_data.apply(create_query, axis=1)
sample_data['latitude'] = np.nan
sample_data['longitude'] = np.nan
for amenity in target_amenities:
    sample_data[amenity] = "No"

# Main loop: Geolocation + POI
for idx, row in sample_data.iterrows():
    lat, lon = get_geolocation(row['query'])
    sample_data.at[idx, 'latitude'] = lat
    sample_data.at[idx, 'longitude'] = lon
    print(f"[{idx+1}/{len(sample_data)}] {row['query']} -> {lat}, {lon}")

    # Get POI if geolocation exists
    if pd.notna(lat) and pd.notna(lon):
        found_pois = get_poi(lat, lon)
        for amenity in target_amenities:
            if amenity in found_pois:
                sample_data.at[idx, amenity] = "Yes"

    time.sleep(0.01)  # Respect Nominatim/Overpass API limits

# Export to CSV
sample_data.to_csv("output/combined_transaction_geolocation_poi_sample1000.csv", index=False, encoding='utf-8-sig')
print("Export complete: combined_transaction_geolocation_poi_sample1000.csv")
