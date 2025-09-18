#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

# Load the data
data = pd.read_csv('C:/Users/Ch. Hamza/Downloads/index_1.csv')

# Inspect the first few rows to understand the structure
print(data.head())

# Check the shape of the dataset (rows, columns)
print(data.shape)

# Get a quick overview of data types and missing values
print(data.info())

df = pd.DataFrame(data)
missing_card = data['card'].isnull().sum()
print(f"Missing values in 'card' column: {missing_card}")

df_cleaned = df.dropna(subset=['card'])

# Verify that missing values are dropped
print(df_cleaned.isnull().sum())

df_clean = df.drop(columns=["date", "datetime", "card"])

# Show first few rows
print(df_clean.head())

from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le_cash = LabelEncoder()
df_clean["cash_type"] = le_cash.fit_transform(df_clean["cash_type"])

le_coffee = LabelEncoder()
df_clean["coffee_name"] = le_coffee.fit_transform(df_clean["coffee_name"])

print(df_clean.head())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Define Features and Target
X = df_clean.drop(columns=["money"])  # Features
y = df_clean["money"]                 # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=5, random_state=42)

# Train (fit) the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

print(le_cash.classes_)    # shows ['card', 'cash'] or ['cash', 'card']
print(le_coffee.classes_)  # shows ['Americano', 'Latte', 'Mocha', ...]

def predict_spending(payment_type, coffee_type):
    # string → number (LabelEncoder ka kaam)
    pay = le_cash.transform([payment_type])[0]        # cash → 1
    coffee = le_coffee.transform([coffee_type])[0]    # Latte → 7

    # model prediction
    pred = model.predict([[pay, coffee]])[0]

    return round(pred, 2)

print("Cash + Latte =", predict_spending("cash", "Latte"))

predicted = model.predict([[le_cash.transform(["cash"])[0],
                            le_coffee.transform(["Latte"])[0]]])
print("Model Predicted:", predicted)

# Ab dataset se real values nikalna
real_values = df[(df["cash_type"] == le_cash.transform(["cash"])[0]) &
                 (df["coffee_name"] == le_coffee.transform(["Latte"])[0])]["money"]

print("Real Values in Dataset:", list(real_values))


# In[9]:


import joblib

# Save the model
joblib.dump(model, "coffee_price_predictor.pkl")

print("Model saved successfully!")


# In[ ]:




