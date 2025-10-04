import pandas as pd
data = pd.read_csv('C:/Users/Ch. Hamza/Downloads/index_1.csv')
print(data.head())
print(data.shape)
print(data.info())

df = pd.DataFrame(data)
missing_card = data['card'].isnull().sum()
print(f"Missing values in 'card' column: {missing_card}")

df_cleaned = df.dropna(subset=['card'])
print(df_cleaned.isnull().sum())

df_clean = df.drop(columns=["date", "datetime", "card"])

print(df_clean.head())

from sklearn.preprocessing import LabelEncoder

le_cash = LabelEncoder()
df_clean["cash_type"] = le_cash.fit_transform(df_clean["cash_type"])

le_coffee = LabelEncoder()
df_clean["coffee_name"] = le_coffee.fit_transform(df_clean["coffee_name"])

print(df_clean.head())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
X = df_clean.drop(columns=["money"])  # Features
y = df_clean["money"]                 # Target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=5, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

print(le_cash.classes_)    
print(le_coffee.classes_)  

def predict_spending(payment_type, coffee_type):
    pay = le_cash.transform([payment_type])[0]       
    coffee = le_coffee.transform([coffee_type])[0]    
    pred = model.predict([[pay, coffee]])[0]

    return round(pred, 2)

print("Cash + Latte =", predict_spending("cash", "Latte"))

predicted = model.predict([[le_cash.transform(["cash"])[0],
                            le_coffee.transform(["Latte"])[0]]])
print("Model Predicted:", predicted)

real_values = df[(df["cash_type"] == le_cash.transform(["cash"])[0]) &
                 (df["coffee_name"] == le_coffee.transform(["Latte"])[0])]["money"]

print("Real Values in Dataset:", list(real_values))

import joblib
joblib.dump(model, "coffee_price_predictor.pkl")

print("Model saved successfully!")






