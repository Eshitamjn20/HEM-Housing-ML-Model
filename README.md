# HEM-Housing Price Prediction-ML-Model
## Table of Contents: :house_with_garden:
01. [Project Introduction](#intro)
02. [Data Source](#source)
03. [Configuring the project Environment](#env)


<a name="intro"></a>
## Project Introduction : 
Finding suitable housing within a budget can be challenging for buyers, and predicting housing prices helps them plan their finances while also benefiting property investors by providing insights into market trends. Residential housing is crucial for individuals and families, making it essential to track property price fluctuations that impact homebuyers, agents, sellers, and investors. This project uses Machine Learning (ML) to predict Melbourne housing prices based on factors like rooms, bathrooms, distance from the CBD, and nearby facilities.


<a name="source"></a>
## Data Source: 
This project leverages two key datasets.  The primary dataset, sourced from Kaggle, consists of real estate data scraped from Domain.com.au, which includes details such as address, property type, suburb, selling method, number of rooms, price, real estate agent, sale date, and distance from the CBD. Although the dataset was raw, we cleaned and processed it to focus on key features that enhance the accuracy of our machine learning models for predicting housing prices. Additionally, we incorporated a schooling dataset to calculate the number of schools per suburb, providing a crucial factor in analyzing regional desirability and its effect on property prices.
- [Kaggle: Melbourne Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)

<a name = "env"></a>
## Configuring the project Environment:
- Creating the Conda Environment:
conda create --name housing-price-env python=3.12.6
- Activate the environment 
conda activate housing-price-env
- Installing the necessary dependencies:  
You can install the required libraries either manually or via a requirements.txt file. The following are the main libraries and versions used:
  - pandas (for data handling)
  - numpy (for numerical operations)
  - matplotlib (for visualizations)
  - seaborn (for visualizations)
  - scikit-learn (for machine learning models)
  - scipy 


- ## Option 1 : Installing the dependencies manually 

    conda install pandas numpy matplotlib seaborn scikit-learn scipy
- ## Option 2 : Using a requirements.txt file

    pip install -r requirements.txt


