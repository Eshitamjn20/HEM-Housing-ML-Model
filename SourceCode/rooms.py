import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the dataset
file_path = '/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Innovation/code/Updated_Final_Cleaned_Melbourne_Housing_Data-3.csv'
data = pd.read_csv(file_path)

##########################################
# 1. Price vs Distance for 5-Room Houses #
##########################################

filtered_data_5_rooms = data[data['Rooms'] == 5]
X_5_rooms = filtered_data_5_rooms['Distance'].values.reshape(-1, 1)
y_5_rooms = filtered_data_5_rooms['Price'].values
log_X_5_rooms = np.log(X_5_rooms)

regressor = LinearRegression()
regressor.fit(log_X_5_rooms, y_5_rooms)
y_pred_5_rooms = regressor.predict(log_X_5_rooms)
r2_5_rooms = r2_score(y_5_rooms, y_pred_5_rooms)

X_range_5_rooms = np.linspace(X_5_rooms.min(), X_5_rooms.max(), 100).reshape(-1, 1)
log_X_range_5_rooms = np.log(X_range_5_rooms)
y_range_pred_5_rooms = regressor.predict(log_X_range_5_rooms)

plt.figure(figsize=(10, 6))
plt.scatter(X_5_rooms, y_5_rooms, alpha=0.6, label="Data points")
plt.plot(X_range_5_rooms, y_range_pred_5_rooms, color='red', label=f"Log fit (R² = {r2_5_rooms:.2f})")
plt.title("Price vs Distance for 5-Room Houses with Log Fit")
plt.xlabel("Distance (km)")
plt.ylabel("Price (AUD)")
plt.legend()
plt.show()

###########################################
# 2. Price vs Distance for 4-Bathroom Houses #
###########################################

filtered_data_4_bathrooms = data[data['Bathroom'] == 4]
X_4_bathrooms = filtered_data_4_bathrooms['Distance'].values.reshape(-1, 1)
y_4_bathrooms = filtered_data_4_bathrooms['Price'].values

poly = PolynomialFeatures(degree=2)
X_poly_4_bathrooms = poly.fit_transform(X_4_bathrooms)

regressor.fit(X_poly_4_bathrooms, y_4_bathrooms)

X_range_4_bathrooms = np.linspace(X_4_bathrooms.min(), X_4_bathrooms.max(), 100).reshape(-1, 1)
X_range_poly_4_bathrooms = poly.transform(X_range_4_bathrooms)
y_range_pred_4_bathrooms = regressor.predict(X_range_poly_4_bathrooms)

r2_4_bathrooms = r2_score(y_4_bathrooms, regressor.predict(X_poly_4_bathrooms))

plt.figure(figsize=(10, 6))
plt.scatter(X_4_bathrooms, y_4_bathrooms, alpha=0.6, label="Data points")
plt.plot(X_range_4_bathrooms, y_range_pred_4_bathrooms, color='red', label=f"Polynomial fit (R² = {r2_4_bathrooms:.2f})")
plt.title("Price vs Distance for 4-Bathroom Houses with Polynomial Fit")
plt.xlabel("Distance (km)")
plt.ylabel("Price (AUD)")
plt.legend()
plt.show()

####################################################
# 3. Price vs Distance for Houses with 500 sqm Land Size #
####################################################

filtered_data_land_size = data[(data['Landsize'] >= 490) & (data['Landsize'] <= 510)]
X_land_size = filtered_data_land_size['Distance'].values.reshape(-1, 1)
y_land_size = filtered_data_land_size['Price'].values

X_poly_land_size = poly.fit_transform(X_land_size)
regressor.fit(X_poly_land_size, y_land_size)

X_range_land_size = np.linspace(X_land_size.min(), X_land_size.max(), 100).reshape(-1, 1)
X_range_poly_land_size = poly.transform(X_range_land_size)
y_range_pred_land_size = regressor.predict(X_range_poly_land_size)

r2_land_size = r2_score(y_land_size, regressor.predict(X_poly_land_size))

plt.figure(figsize=(10, 6))
plt.scatter(X_land_size, y_land_size, alpha=0.6, label="Data points")
plt.plot(X_range_land_size, y_range_pred_land_size, color='red', label=f"Polynomial fit (R² = {r2_land_size:.2f})")
plt.title("Price vs Distance for Houses with Land Size around 500 sqm")
plt.xlabel("Distance (km)")
plt.ylabel("Price (AUD)")
plt.legend()
plt.show()
