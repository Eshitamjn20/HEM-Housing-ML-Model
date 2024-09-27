HEM-Housing Price Prediction-ML-Model

Table of Contents :house_with_garden:

1. Project Introduction
2. Data Source
3. Configuring the Project Environment
4. Data Visualization and Exploration
5. Feature Engineering and Model Training
6. Visualizing Price and Feature Relationships
7. Model Evaluation
8. How to Use the Model for Predictions
<a name="intro"></a>

1. Project Introduction

Finding suitable housing within a budget can be challenging for buyers, and predicting housing prices helps them plan their finances while also benefiting property investors by providing insights into market trends. Residential housing is crucial for individuals and families, making it essential to track property price fluctuations that impact homebuyers, agents, sellers, and investors. This project uses Machine Learning (ML) to predict Melbourne housing prices based on factors like rooms, bathrooms, distance from the CBD, and nearby facilities.

<a name="source"></a>

2. Data Source

This project leverages two key datasets. The primary dataset, sourced from Kaggle, consists of real estate data scraped from Domain.com.au, which includes details such as address, property type, suburb, selling method, number of rooms, price, real estate agent, sale date, and distance from the CBD. The secondary dataset includes schooling facilities, providing additional information on nearby schools, an important factor influencing property prices.

Kaggle: Melbourne Housing Market
<a name="env"></a>

3. Configuring the Project Environment

Conda Environment Setup:
Creating the Conda Environment:
bash
Copy code
conda create --name housing-price-env python=3.12.6
Activate the Environment:
bash
Copy code
conda activate housing-price-env
Installing the Necessary Dependencies:
You can install the required libraries either manually or via a requirements.txt file.
Option 1: Installing the Dependencies Manually
bash
Copy code
conda install pandas numpy matplotlib seaborn scikit-learn scipy
Option 2: Using the requirements.txt file
bash
Copy code
pip install -r requirements.txt
<a name="eda"></a>

4. Data Visualization and Exploration

Once the environment is set up, you can run the EDA (Exploratory Data Analysis) and visualize the dataset using the eda.py script.

To explore and visualize data, run:
bash
Copy code
python scripts/eda.py
The script performs the following:

Checks for missing values.
Generates univariate and bivariate plots to explore data distribution and relationships.
Produces a correlation heatmap for key features like Rooms, Bathrooms, Car_Spot, Schooling_Facilities, and Price.
Example of Generated Plots:

Price Distribution
Price vs Rooms
Price vs Bathrooms
Price vs Car Spots
Price vs Property Type
Price vs Region
<a name="model"></a>

5. Feature Engineering and Model Training

In this step, we create new features based on existing ones (e.g., interaction features) and train three models:

Random Forest Regressor (with Hyperparameter Tuning)
Polynomial Regression (Degree 2)
Gradient Boosting Regressor
These models are trained to predict housing prices based on the cleaned and engineered features.

To train the models, run:
bash
Copy code
python scripts/feature_engineering.py
The script does the following:

Performs Feature Engineering:
Adds features like Rooms_Bathroom_Interaction, Schools_Distance_Ratio, etc.
Splits the Dataset:
Uses an 80%/20% train-test split to evaluate model performance.
Trains the Models using RandomForestRegressor, PolynomialFeatures, GradientBoostingRegressor.
<a name="relations"></a>

6. Visualizing Price and Feature Relationships

We also visualize how certain features impact housing prices using Polynomial Regression. This includes:

Price vs Distance from CBD for 5-room houses.
Price vs Distance from CBD for 3-bathroom houses.
Price vs Distance including the number of nearby schools.
To generate relationship plots, run:
bash
Copy code
python scripts/relations.py
The script generates visualizations showing how different factors affect housing prices, helping to understand the relationships between distance, schools, rooms, bathrooms, and price.

<a name="evaluation"></a>

7. Model Evaluation

To evaluate the model's performance, the script prints key metrics for each model:

R² (R-squared): How well the model explains variance in the data.
MSE (Mean Squared Error): The average squared difference between actual and predicted values.
After training and evaluating the models, predictions are compared against actual prices.

To evaluate models, run:
bash
Copy code
python scripts/feature_engineering.py
Example Output:

bash
Copy code
Random Forest (Tuned) Performance:
Train R²: 0.95
Test R²: 0.82
Test MSE: 1234567.89
<a name="usage"></a>

8. How to Use the Model for Predictions

Once trained, the models can be used to predict housing prices for new data.

How to Use the Models:
Ensure the Conda environment is activated:
bash
Copy code
conda activate housing-price-env
Run the model prediction script:
bash
Copy code
python scripts/feature_engineering.py
Pass in New Data:
To predict prices for new properties, ensure the new data matches the format of the dataset used in training (including features like rooms, bathrooms, distance, schooling facilities).
