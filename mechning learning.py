#!/usr/bin/env python
# coding: utf-8

# In[23]:


import requests 
import pandas as pd
import os       
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# In[24]:


df = pd.read_csv("1111ניקוי מעודכן סופי")


# In[25]:


regressor=LinearRegression()

X=df[df.columns[df.columns != 'Price']]
Y=df[['Price']]
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 ,random_state=42 )

regressor.fit(X_train , Y_train)

Y_pred=regressor.predict(X_test)


# In[26]:


sns.regplot(Y_test, Y_pred, scatter_kws={'color':'red'}, line_kws={'color':'black'})


# In[27]:


mse= mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
rmse


# In[28]:


pred=regressor.score(X_test,Y_test)
print(pred)


# In[29]:


df.info()


# In[30]:


X=df[df.columns[df.columns != 'Price']]
Y=df[['Price']]
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 ,random_state=42 )

regressor.fit(X_train , Y_train)
Y_pred=regressor.predict(X_test)


# In[31]:


# Split the data into features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest regression model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


# In[32]:


# Split the data into features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regression model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


# In[33]:


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# assuming your data is in a pandas DataFrame called df
X = df.drop('Price', axis=1) # features
y = df['Price'] # target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(X.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the model
model.fit(X_train, y_train, epochs=10)

# make predictions on the test set
y_pred = model.predict(X_test)

# calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R-squared: {r2:.2f}')


# In[34]:


# Split the data into features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


# In[35]:


#random Forest:
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, Y_train)
print(regressor.score(X_test, Y_test))


# In[36]:


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# assuming your data is in a pandas DataFrame called df
X = df.drop('Price', axis=1) # features
y = df['Price'] # target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(X.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the model
model.fit(X_train, y_train, epochs=10)

# make predictions on the test set
y_pred = model.predict(X_test)

# calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R-squared: {r2:.2f}')


# In[37]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# assuming your data is in a pandas DataFrame called df
X = df.drop('Price', axis=1) # features
y = df['Price'] # target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create and fit the model
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the results
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')

print(f'R-squared: {r2:.2f}')


# In[38]:


# Split the data into features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


# In[ ]:





# In[ ]:





# In[ ]:




