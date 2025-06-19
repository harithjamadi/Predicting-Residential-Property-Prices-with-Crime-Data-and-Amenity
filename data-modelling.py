import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Use Seaborn for clean gridlines and style
sns.set(style="whitegrid")

# =========================
# 📂 Load and Clean Dataset
# =========================

data = pd.read_csv("output/combined_transaction_geolocation_poi_sample1000.csv")

# Clean price and numeric columns
data['Transaction Price'] = data['Transaction Price  '].replace({'RM': '', ',': ''}, regex=True).astype(float)
data['Main Floor Area'] = pd.to_numeric(data['Main Floor Area'], errors='coerce').fillna(0)
data['Land/Parcel Area'] = pd.to_numeric(data['Land/Parcel Area'], errors='coerce').fillna(0)

# Encode amenities as 1/0
amenity_columns = ['school', 'kindergarten', 'university', 'hospital', 'clinic',
                   'supermarket', 'place_of_worship', 'bus_station', 'marketplace']
for col in amenity_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Features and target
features = ['Main Floor Area', 'Land/Parcel Area'] + amenity_columns
X = data[features]
y = data['Transaction Price']

# ====================
# 📂 Train-test split
# ====================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): RM{mae:,.2f}")
print(f"R² Score: {r2:.3f}")

# ============================
# 📂 Scatter Plot – Purple Theme
# ============================

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, color='#800080', alpha=0.7, edgecolor='k', s=70, label="Predicted Points")

# Diagonal line for perfect prediction
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label="Perfect Prediction")

# Labels and aesthetics
plt.xlabel("Actual Transaction Price (RM)", fontsize=12)
plt.ylabel("Predicted Transaction Price (RM)", fontsize=12)
plt.title(f"Scatter Plot: Actual vs Predicted Transaction Prices\n(RandomForest, MAE = RM{mae:,.2f}, R² = {r2:.3f})", fontsize=13, weight='bold')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
