# HEM-Housing Price Prediction-ML-Model

## Table of Contents :house_with_garden:

01. [Project Introduction](#intro)
02. [Data Source](#source)
03. [Configuring the Project Environment](#env)
04. [Data Processing](#data-processing)
05. [Model Training](#model-training)
06. [Model Prediction](#model-prediction)
07. [Evaluation Metrics](#evaluation)


<a name="intro"></a>

## Project Introduction

Finding suitable housing within a budget can be challenging for buyers, and predicting housing prices helps them plan their finances while also benefiting property investors by providing insights into market trends. Residential housing is crucial for individuals and families, making it essential to track property price fluctuations that impact homebuyers, agents, sellers, and investors. This project uses Machine Learning (ML) to predict Melbourne housing prices based on factors like rooms, bathrooms, distance from the CBD, and nearby facilities.

<a name="source"></a>

## Data Source

This project leverages two key datasets. The primary dataset, sourced from Kaggle, consists of real estate data scraped from Domain.com.au, which includes details such as address, property type, suburb, selling method, number of rooms, price, real estate agent, sale date, and distance from the CBD. We also incorporated a schooling dataset to calculate the number of schools per suburb, providing a crucial factor in analyzing regional desirability and its effect on property prices.

Kaggle: Melbourne Housing Market
<a name="env"></a>

Configuring the Project Environment

To set up the environment for this project, follow these steps:

Step 1: Creating the Conda Environment
bash
Copy code
conda create --name housing-price-env python=3.8
Step 2: Activate the environment
bash
Copy code
conda activate housing-price-env
Step 3: Install the necessary dependencies
You can install the required libraries either manually or via a requirements.txt file.

Option 1: Installing dependencies manually

bash
Copy code
conda install pandas numpy matplotlib seaborn scikit-learn scipy
Option 2: Using a requirements.txt file

bash
Copy code
pip install -r requirements.txt
<a name="data-processing"></a>

## Data Processing

To explore and process the dataset, we perform Exploratory Data Analysis (EDA) and feature engineering. This step involves analyzing data distributions, handling missing data, and creating interaction terms between features like rooms, bathrooms, and schools.

## EDA Script:
To perform EDA, run the Eda.py script:

bash
Copy code
python scripts/Eda.py
This script will:

Load the housing dataset.
Generate various visualizations, including distribution plots and correlation heatmaps.
Clean the dataset by removing missing values.
Feature Engineering:
Feature engineering involves creating new features based on existing data, which is handled in the feature_engineering.py script. Key features include:

Schools_Distance_Ratio: Ratio of the number of schools to the distance from the CBD.
Rooms_Bathroom_Interaction: Interaction term between rooms and bathrooms.
Distance_Schools_Interaction: Interaction term between distance and schooling facilities.
To run the feature engineering script, use:

bash
Copy code
python scripts/feature_engineering.py
<a name="model-training"></a>

## Model Training

We use three primary machine learning models for housing price prediction:

Random Forest Regressor (with hyperparameter tuning)
Polynomial Regression
Gradient Boosting Regressor
Training the Models:
All models are trained in the feature_engineering.py script. To train the models and evaluate their performance:

bash
Copy code
python scripts/feature_engineering.py
The script performs the following steps:

Splits the dataset into training and testing sets.
Trains three models with cross-validation and hyperparameter tuning for Random Forest.
Evaluates each model using metrics like R² and Mean Squared Error (MSE).
Saves the trained models.
Hyperparameter Tuning:
For the Random Forest model, we apply GridSearchCV to find the optimal hyperparameters (e.g., the number of trees, depth of trees, etc.). The best hyperparameters are automatically selected based on model performance.

<a name="model-prediction"></a>

## Model Prediction

Once the models are trained, they can be used to predict housing prices for new data.

Making Predictions:
You can use the trained models for predictions directly within the feature_engineering.py script. For example, you can predict prices using the test data, and visualize actual vs. predicted prices for each model:

bash
Copy code
python scripts/feature_engineering.py --predict --data new_data.csv
This will output the predicted prices along with the actual prices in a tabular format.

## Visualizations:
The script generates visualizations showing how well each model's predictions align with actual prices:

Scatter plots of actual vs. predicted prices for each model.
Residual plots for error analysis.
<a name="evaluation"></a>

## Evaluation Metrics

We evaluate the models using the following metrics:

R² (Coefficient of Determination): Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
Mean Squared Error (MSE): The average squared difference between the actual and predicted values.
The performance of each model is printed in the console after training, showing R² and MSE values for both the training and testing datasets.
