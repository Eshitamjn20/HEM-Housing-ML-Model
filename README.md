# HEM-Housing Price Prediction-ML-Model

## Table of Contents :house_with_garden:

1. [Project Introduction](#project-introduction)
2. [Data Source](#data-source)
3. [Configuring the Project Environment](#configuring-the-project-environment)
4. [Running the Jupyter Notebooks](#running-the-jupyter-notebooks)
5. [Data Processing (EDA)](#eda)
6. [Relationships Visualization](#visualise)
7. [Feature Engineering and Model Training](#feature-engineering-and-model-training)
8. [Model Prediction](#model-prediction)

<a name="project-introduction"></a>
## Project Introduction

Predicting housing prices is vital for buyers, sellers, and investors. In this project, we predict Melbourne housing prices based on factors such as the number of rooms, bathrooms, proximity to the Central Business District (CBD), and nearby school facilities. We use machine learning models such as Random Forest, Gradient Boosting, and Polynomial Regression to achieve this.

---
<a name="data-source"></a>
## Data Source

- **[Kaggle: Melbourne Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)**  
This dataset includes a variety of features, including property type, suburb, number of rooms, price, sale date, and distance from the CBD.
  
- **[Additional Data: Schools per Suburb](https://discover.data.vic.gov.au/dataset/school-locations-2021)**  
We enriched the dataset with schooling data from the Victorian Government dataset, which includes the number of schools per suburb. This is important to assess how school proximity influences property prices.

---
<a name="configuring-the-project-environment"></a>
## Configuring the Project Environment

### Step 1: Clone the Repository using the following commands 
git clone https://github.com/Eshitamjn20/HEM-Housing-ML-Model.git
cd your-repository //folder where the repository was clonned 

### Step 2: Install Anaconda
Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

### Step 3: Create a Conda Environment

Create and activate a new environment using Conda

conda create --name housing-price-env python=3.12.6
conda activate housing-price-env

### Step 3: Install Required Dependencies  
You can either manually install the libraries or use the `requirements.txt` file.

#### Option 1: Manually Install Dependencies


 conda install pandas numpy matplotlib seaborn scikit-learn scipy


#### Option 2: Using `requirements.txt` file

conda install --file requirements.txt


## Running the Jupyter Notebooks

Once your environment is set up, you can run the Jupyter notebooks in sequence. Each notebook is designed to handle a specific part of the project, including data exploration, visualizations, feature engineering, model training, and evaluation.

### Step-by-Step Process:
- Activate the Environment Make sure your conda environment is activated:

conda activate housing-price-env

- Launch Jupyter Notebook
   jupyter notebook
- Open the Notebooks The following notebooks will appear in the Jupyter interface:
 **melb-housing-dataset-cleaning-eda.ipynb**
- This notebook explores the dataset and performs Exploratory Data Analysis (EDA) to understand the features and relationships in the dataset.
 **housing_price_regression_filtered_analysis.ipynb**
- This notebook generates key visualizations to show relationships between various features and housing prices.
 **Modelling-and-feature-engineering.ipynb**
- This notebook performs feature engineering, trains machine learning models, and evaluates their performance.
<a name="eda"></a>

## Data Processing (EDA)

<<<<<<< HEAD
Notebook: melb-housing-dataset-cleaning-eda.ipynb
=======
Notebook: [EDA] (https://github.com/Eshitamjn20/HEM-Housing-ML-Model/blob/main/SourceCode/melb-housing-dataset-cleaning-eda.ipynb)
>>>>>>> 99db115d20a2595ad6cc89a89d1cdb4dc7ca4c9d
### Steps:

### Loading the Dataset
The dataset is loaded, and the first few rows are displayed to provide an overview of the data.
### Handling Missing Values
The notebook checks for missing values and removes any rows with missing data, ensuring data consistency.
### Univariate Analysis
You will visualize the distribution of individual features such as Price, Distance_from_CBD, Rooms, and Bathrooms. Histograms are used to show these distributions.
### Bivariate Analysis
This step involves analyzing the relationships between pairs of features, for example:
- Price vs Rooms
- Price vs Bathrooms
- Price vs Distance from CBD

- The bivariate plots (box plots and scatter plots) will help reveal correlations between these features and the house prices.
### Output:

- Heatmaps: Visualizes the correlation between the features.
- Histograms: Displays the distributions of key variables.
- Box Plots: Shows how housing price varies by number of rooms, bathrooms, etc.
<a name="visualise"></a>

## Relationships Visualization

### Notebook: housing_price_regression_filtered_analysis.ipynb
Steps:

- Price vs Distance for 5-Room Houses
A scatter plot showing how housing prices vary with the distance from the CBD for properties that have 5 rooms. This is fitted with a polynomial regression line to capture the trend.
- Price vs Distance for 3-Bathroom Houses
Another scatter plot showing how housing prices vary with the distance from the CBD, but this time for properties with 3 bathrooms.
- Price vs Distance with Number of Schools
A scatter plot showing how the number of nearby schools and the distance from the CBD affect housing prices.

### Output:

- Scatter Plots with Regression Lines: Visualizes relationships between prices and key variables like rooms, bathrooms, and distance from CBD with nearby schools.
- R-squared Value: This value helps to evaluate how well the regression line fits the data.
<a name="feature-engineering-and-model-training"></a>

## Feature Engineering and Model Training

### Notebook: Modelling-and-feature-engineering.ipynb
Steps:

Feature Creation
- New features are engineered to help improve the accuracy of predictions:
= Schools_Distance_Ratio: Ratio of schools to the distance from the CBD.
- Rooms_Bathroom_Interaction: Interaction term between the number of rooms and bathrooms.
- Distance_Schools_Interaction: Interaction between the distance from the CBD and nearby schools.
### Model Training
Three different machine learning models are trained:
- Random Forest: A powerful model that captures complex relationships between features.
- Polynomial Regression: Adds polynomial terms to model non-linear relationships.
- Gradient Boosting: Captures complex relationships and improves accuracy by sequentially learning from previous predictions.
### Model Evaluation
Each model is evaluated using the following metrics:
- R² (Coefficient of Determination): Measures how well the model captures variance in housing prices.
- MSE (Mean Squared Error): Measures the average squared difference between actual and predicted prices.

### Output:

- Best Hyperparameters: After hyperparameter tuning (using GridSearchCV), the best parameters for Random Forest are shown.
- Predictions: Each model makes predictions on test data, which are then compared with actual prices.
- Scatter Plots: Visualizes how well the models' predictions align with actual prices.
- Comparison Table: Displays the actual prices vs predicted prices for each model.

<a name="model-prediction"></a>

### Model Prediction

After running the FeatureEngineering.ipynb notebook, you will have trained three models: Random Forest, Polynomial Regression, and Gradient Boosting. The notebook visualizes and compares their predictions with actual prices.

### Outputs to Expect:

- Scatter Plots: Reduced scatter plots to make the predictions easier to compare. These plots will help you visualize how closely each model's predictions match the actual prices.
- Comparison Table: A table is displayed showing the actual prices vs the predicted prices for each of the models (Random Forest, Polynomial Regression, and Gradient Boosting).
