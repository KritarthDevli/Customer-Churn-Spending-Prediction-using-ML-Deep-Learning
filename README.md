Customer Churn Spending Prediction using Machine Learning & Deep Learning
This project focuses on predicting customer total charges using both classical machine learning and deep learning approaches.
It demonstrates the complete end-to-end workflow of preparing real-world data, handling inconsistencies, preprocessing features, and building regression models to evaluate which method performs better.

📂 Project Overview
Telecom companies rely heavily on customer analytics to understand user behaviour, predict churn, and estimate lifetime value.
In this project, we analyse a real telecom dataset containing customer demographics, subscription details, service usage, and churn information. The objective is to:

🎯 Goal
Predict the totalCharges (customer spending) using:
Linear Regression (Scikit-Learn)
Deep Learning Neural Network (Keras/TensorFlow)

🧠 Key Learning Outcomes
By completing this project, the following skills are demonstrated:
Handling dirty, real-world datasets
Detecting missing values and inconsistencies
Preprocessing categorical and numerical variables
One-hot encoding & scaling features
Building ML models using Scikit-Learn
Building and training neural networks using Keras
Comparing model performance using MSE
Understanding regression workflows used in AI & business analytics

📊 Dataset Description
The dataset contains customer information including:

Feature	Description
gender	Customer gender
SeniorCitizen	Whether the customer is elderly
Partner	Customer lives with partner
Dependents	Children/dependents
tenure	Total months with company
PhoneService	Landline service
InternetService	DSL / Fiber / None
Contract	Month-to-month / One year / Two year
PaymentMethod	Payment mode
monthlyCharges	Monthly bill amount
totalCharges	Target variable – total customer spending

The dataset initially contains missing values, inconsistencies, and categorical variables that require cleaning before modelling.

🛠️ Tech Stack
Python 3.10+
Pandas – data manipulation
NumPy – numerical operations
Matplotlib / Seaborn – data visualisation
Scikit-Learn – preprocessing & linear regression
TensorFlow / Keras – deep learning model
Jupyter Notebook – development environment

🔧 Project Workflow

1️⃣ Load & Inspect Data

Display head, info, summary statistics
Identify missing values
Understand structure and data types

2️⃣ Initial Preprocessing

Drop duplicates
Remove missing values
One-hot encode categorical features
Select features and target
Train/test split

3️⃣ Numerical Feature Scaling

Standardisation using StandardScaler to improve model performance.

4️⃣ Machine Learning Model

A Linear Regression model is trained and evaluated using MSE.

5️⃣ Deep Learning Model

A Keras Sequential Model with:
Dense(64, ReLU)
Dense(32, ReLU)
Dense(1)

6️⃣ Model Comparison

Both models are compared using Mean Squared Error (MSE) to determine which performs better.
