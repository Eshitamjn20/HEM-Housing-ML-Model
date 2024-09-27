import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import mstats

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Reading the cleaned dataset from the provided file path
file_path = '/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Innovation/code/mel-housing-data-cleaned.csv'
mel_housing_df = pd.read_csv(file_path)

# Exploring the dataset
print(f"Number of rows in the dataset: {mel_housing_df.shape[0]}")
print(f"Number of columns in the dataset: {mel_housing_df.shape[1]}\n")
mel_housing_df.info()

# Checking for missing data in the cleaned dataset
missing_data = mel_housing_df.isnull().sum()
if missing_data.sum() == 0:
    print("\nThere are no missing values in the dataset.")
else:
    print("\nThe dataset has the following missing values:")
    print(missing_data[missing_data > 0])

# Generating heatmap for finalized features
columns_to_plot = ['Price', 'Distance_from_CBD', 'Rooms', 'Bathroom', 'Car_Spot', 'Schooling_Facilities']
plt.figure(figsize=(8,6))
sns.heatmap(mel_housing_df[columns_to_plot].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Final Features')
plt.show()

# Univariate Analysis (Distribution of individual variables)
plt.figure(figsize=(15, 5))

# Price Distribution
plt.subplot(2, 3, 1)
sns.histplot(mel_housing_df['Price'], bins=30, kde=True)
plt.title('Price Distribution')

# Distance from CBD Distribution
plt.subplot(2, 3, 2)
sns.histplot(mel_housing_df['Distance_from_CBD'], bins=30, kde=True)
plt.title('Distance from CBD Distribution')

# Rooms Distribution
plt.subplot(2, 3, 3)
sns.histplot(mel_housing_df['Rooms'], bins=10, kde=True)
plt.title('Rooms Distribution')

# Bathrooms Distribution
plt.subplot(2, 3, 4)
sns.histplot(mel_housing_df['Bathroom'], bins=10, kde=True)
plt.title('Bathrooms Distribution')

# Car_Spot Distribution
plt.subplot(2, 3, 5)
sns.histplot(mel_housing_df['Car_Spot'], bins=10, kde=True)
plt.title('Car Spot Distribution')

# Schooling Facilities Distribution
plt.subplot(2, 3, 6)
sns.histplot(mel_housing_df['Schooling_Facilities'], bins=10, kde=True)
plt.title('Schools Distribution')

plt.tight_layout()
plt.show()

# Bivariate Analysis (Relationships between two variables)
# Price vs Number of Rooms (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(x=mel_housing_df['Rooms'], y=mel_housing_df['Price'])
plt.title('Price vs Rooms')
plt.xlabel('Number of Rooms')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Price vs Number of Bathrooms (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(x=mel_housing_df['Bathroom'], y=mel_housing_df['Price'])
plt.title('Price vs Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Price vs Car Spots (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(x=mel_housing_df['Car_Spot'], y=mel_housing_df['Price'])
plt.title('Price vs Car Spots')
plt.xlabel('Number of Car Spots')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Plot Price against Rooms with Bathroom as the hue
plt.figure(figsize=(15, 8))
sns.boxplot(x="Rooms", y="Price", hue="Bathroom", data=mel_housing_df)
plt.title('Price vs Rooms and Bathrooms')
plt.xlabel('Number of Rooms')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Box plot for Price vs Property Type
plt.figure(figsize=(10, 6))
sns.boxplot(x="Type", y="Price", data=mel_housing_df)
plt.title('Price vs Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Boxplot for Region and Price
plt.figure(figsize=(8,4))
sns.boxplot(x="Region", y="Price", data=mel_housing_df)
plt.title('Price vs Region')
plt.xlabel('Region')
plt.ylabel('Price (AUD)')
plt.xticks(rotation=45)  
plt.grid(True)
plt.show()

# Boxplot for Schooling Facilities and Price
plt.figure(figsize=(10, 6))
sns.boxplot(x="Schooling_Facilities", y="Price", data=mel_housing_df)
plt.title('Schooling Facilities vs Price')
plt.xlabel('Number of Schooling Facilities')
plt.ylabel('Price (AUD)')
plt.grid(True)
plt.show()

# Saving the final cleaned dataset
output_path = '/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Innovation/code/mel-housing-data-cleaned.csv'
mel_housing_df.to_csv(output_path, index=False)

print("Data processing complete, cleaned dataset saved.")
