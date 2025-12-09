#TASK 1: LOAD AND INSPECT THE DATA 
#-----------------------------------

#Importing all the libraries that are used in the project
import pandas as pd  
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers

#Adding the required columns
cols = ['ID','senior','tenure','streaming','contract','payMethod', 
        'monthlyCharges','totalCharges','churn']

#loaded the dataset in pandas DataFrame
df = pd.read_csv("CustomerChurn.csv") 
df.columns = cols

#printing first 5 rows of dataset
print("First 5 rows of the dataset are :") 
print(df.head())

#checking the shape of dataset
print("\nShape of dataset:", df.shape) 

#checking datatype in dataset
print("\nData Types:") 
print(df.dtypes)

#finding all the missing values in the dataset
print("\nMissing values per column:") 
print(df.isna().sum())

#TASK 2 : INITIAL PREPROCESSING (BASIC CLEANING) & SPLIT
#---------------------------------------------------------

#handling  values by Converting totalCharges to numeric
df['totalCharges'] = pd.to_numeric(df['totalCharges'], errors='coerce') 

#Drop rows where the target variable (totalCharges) is missing.
df = df.dropna(subset=['totalCharges']) 

#dropped column ID as it was irrevalent for the project
df = df.drop(columns=['ID']) 

#handling  values by Converting monthlyCharges to numeric
df['monthlyCharges'] = pd.to_numeric(df['monthlyCharges'], errors='coerce')

#handling  values by Converting tenure to numeric
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

#Defining features (X) and target (y)
X = df.drop(columns=['totalCharges'])
y = df['totalCharges']

#My raptor username is KD13 so used 13 as a random_key
random_key = 13  

#Splitting the data into training(80%) and testing(20%) sets
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=random_key 
)           

#calculating training shape
print("Calculating Training shape:", X_train.shape) 

#calculating test shape
print(" Calculating Test shape:", X_test.shape) 

#TASK 3: FURTHER PREPROCESSING (POST-SPLIT)  
#-------------------------------------------

#Identifying categorical columns
cat_cols = X_train.select_dtypes(include=['object']).columns  

#Identifying numerical columns
num_cols = X_train.select_dtypes(include=['int64','float64']).columns 

#Numeric pipeline
#Changing missing numeric values with the median 
#Scaling numeric features using StandardScaler for ANN normalization
num_pipeline = Pipeline([    
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

#Categorical pipeline
#Calculating missing categorical values using most frequent category
#One-hot encoding categorical features into numeric format
cat_pipeline = Pipeline([ 
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')) 
])

#Combining both numeric and categorical pipelines into ColumnTransformer
preprocess = ColumnTransformer([ 
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

#Fits the preprocessing steps on training set only
X_train_processed = preprocess.fit_transform(X_train)  

#Applying same learned transformations to test set
X_test_processed = preprocess.transform(X_test)

#displaying the processed shapes
print("Processed train shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

#TASK 4: DECISION TREE REGRESSION
#----------------------------------

#Trained DecisionTreeRegressor with training data
tree_model = DecisionTreeRegressor(random_state=random_key) 
tree_model.fit(X_train_processed, y_train)

#pridicted the feature
y_pred_tree = tree_model.predict(X_test_processed) 
r2_tree = r2_score(y_test, y_pred_tree)

#Calulating the R2 score
print("Decision Tree R2:", r2_tree)  

#Calculating predicted vs actual value
plt.figure()
plt.scatter(y_test, y_pred_tree)
plt.xlabel("Actual totalCharges")
plt.ylabel("Predicted totalCharges")
plt.title("Decision Tree: Predicted vs Actual") 
plt.grid(True)
plt.show()

#TASK 5: NEURAL NETWORK REGRESSION (KERAS)  
#------------------------------------------

#Convert to dense arrays for Keras 
if sp.issparse(X_train_processed):  
    X_train_nn = X_train_processed.toarray()
else:
    X_train_nn = X_train_processed

if sp.issparse(X_test_processed):
    X_test_nn = X_test_processed.toarray()
else:
    X_test_nn = X_test_processed

#Added dimension for ANN
input_dim = X_train_nn.shape[1] 

#ANN architecture
#64 & 32 neurons chosen to balance complexity and overfitting
#ReLU for non-linearity
#Linear output (single neuron) for regression
#Building a fully-connected feedforward neural network using Keras
model = keras.Sequential([             
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

#Compiled the ANN using Adam optimizer and MSE loss for regression
model.compile(optimizer='adam', loss='mse')

#Trainning the model
history = model.fit(   
    X_train_nn, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

#Plotting trainning history (loss curve)
plt.figure()  
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training History")
plt.legend()
plt.grid(True)
plt.show()

#Calculated Predictions
y_pred_nn = model.predict(X_test_nn).flatten() 

#Calculated R2 score
r2_nn = r2_score(y_test, y_pred_nn) 
print("Neural Network R2 score is :", r2_nn)

#Finally calculating predicted vs actual values (ANN)
plt.figure()
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.xlabel("Actual totalCharges")
plt.ylabel("Predicted totalCharges")
plt.title("Neural Network: Actual vs Predicted")
plt.grid(True)
plt.show()

#Calculating the r2 score of both decision tree and neural network
print("Decision Tree R2 score is:", r2_tree)
print("Neural Network R2 score is:", r2_nn)
