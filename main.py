import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and clean data
data = pd.read_csv("homes.csv")
data.columns = data.columns.str.replace('"', '').str.strip()

# ML part
X = data.drop(columns=["List"])
y = data["List"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
# 🔢 Predict using custom user input
print("\n🔮 Enter your own house details to predict price:")

living = float(input("Living area (sq.ft): "))
rooms = int(input("Number of rooms: "))
beds = int(input("Number of bedrooms: "))
baths = int(input("Number of bathrooms: "))
age = int(input("Age of the house: "))
acres = float(input("Land size in acres: "))
taxes = int(input("Expected property tax: "))
sell = int(input("Selling price: "))

# Create dataframe for prediction
user_input = pd.DataFrame([[sell, living, rooms, beds, baths, age, acres, taxes]],
                          columns=["Sell", "Living", "Rooms", "Beds", "Baths", "Age", "Acres", "Taxes"])

# Predict the price
predicted_price = model.predict(user_input)[0]
print(f"\n💰 Predicted House Price (List Price): ₹{predicted_price:.2f}")





