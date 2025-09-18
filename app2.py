# trigger redeploy
import streamlit as st
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# --- Load model and encoders ---
model = pickle.load(open("Model1/untitled8.pkl", "rb"))
le_cash = pickle.load(open("Model1/cash_encoder.pkl", "rb"))
le_coffee = pickle.load(open("Model1/coffee_encoder.pkl", "rb"))

st.title("💰 Coffee Spending Predictor")

# Dropdowns using encoder classes
payment_type = st.selectbox("Select Payment Type", le_cash.classes_)
coffee_type = st.selectbox("Select Coffee Type", le_coffee.classes_)

# Encode user input
pay = le_cash.transform([payment_type])[0]
coffee = le_coffee.transform([coffee_type])[0]

# Predict button
if st.button("Predict Spending"):
    pred = model.predict([[pay, coffee]])[0]
    st.success(f"Predicted Spending: {round(pred, 2)} 💵")

# --- Seaborn plot section ---
st.subheader("💹 Spending by Coffee Type and Payment Method")

# Load dataset for plotting
df = pd.read_csv("index_1.csv")


# Clean dataset similar to ML preprocessing
df_clean = df.dropna(subset=['money', 'cash_type'])
df_clean["cash_type"] = df_clean["cash_type"].str.capitalize()
df_clean = df_clean[df_clean["coffee_name"].isin(le_coffee.classes_)]
df_clean = df_clean[df_clean["cash_type"].isin(le_cash.classes_)]

# Plot with Seaborn
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="coffee_name", y="money", hue="cash_type", data=df_clean, ax=ax)
ax.set_xlabel("Coffee Name")
ax.set_ylabel("Spending")
ax.set_title("Average Spending by Coffee & Payment Type")

st.pyplot(fig)



