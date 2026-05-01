import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load model + scaler
model = pickle.load(open("finalized_model1.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Load dataset for EDA
df = pd.read_csv("Churn_Modelling.csv")

st.set_page_config(page_title="Bank Churn App", layout="wide")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Prediction", "📊 EDA Dashboard"]
)

# ======================================================
# 1. PREDICTION PAGE
# ======================================================
if menu == "🏠 Prediction":

    st.title("🏦 Bank Customer Churn Prediction")

    st.write("Enter customer details below:")

    credit_score = st.number_input("Credit Score", 300, 900, 650)
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    num_products = st.number_input("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    if st.button("Predict Churn"):

        input_data = np.array([[credit_score, age, tenure, balance,
                                num_products, has_cr_card,
                                is_active_member, estimated_salary,
                                geo_germany, geo_spain, gender_male]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

# ======================================================
# 2. EDA DASHBOARD
# ======================================================
elif menu == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis Dashboard")

    # =========================
    # 🔥 SIDEBAR FILTERS
    # =========================
    st.sidebar.subheader("🔎 Filter Data")

    geo_filter = st.sidebar.multiselect(
        "Select Geography",
        options=df["Geography"].unique(),
        default=df["Geography"].unique()
    )

    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    churn_filter = st.sidebar.multiselect(
        "Select Churn Status",
        options=df["Exited"].unique(),
        default=df["Exited"].unique()
    )

    age_filter = st.sidebar.slider(
        "Select Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max()))
    )

    # =========================
    # 🎯 APPLY FILTERS
    # =========================
    filtered_df = df[
        (df["Geography"].isin(geo_filter)) &
        (df["Gender"].isin(gender_filter)) &
        (df["Exited"].isin(churn_filter)) &
        (df["Age"].between(age_filter[0], age_filter[1]))
    ]

    # =========================
    # 📊 DATA OVERVIEW
    # =========================
    st.subheader("Filtered Dataset")
    st.write(filtered_df.head())
    st.write("Shape:", filtered_df.shape)

    # =========================
    # 📈 CHURN DISTRIBUTION
    # =========================
    st.subheader("📈 Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Exited", data=filtered_df, ax=ax)
    st.pyplot(fig)

    # =========================
    # 🌍 GEOGRAPHY VS CHURN
    # =========================
    st.subheader("🌍 Geography vs Churn")

    fig, ax = plt.subplots()
    sns.countplot(x="Geography", hue="Exited", data=filtered_df, ax=ax)
    st.pyplot(fig)

    # =========================
    # 👥 GENDER VS CHURN
    # =========================
    st.subheader("👥 Gender vs Churn")

    fig, ax = plt.subplots()
    sns.countplot(x="Gender", hue="Exited", data=filtered_df, ax=ax)
    st.pyplot(fig)

    # =========================
    # 📉 AGE DISTRIBUTION
    # =========================
    st.subheader("📉 Age Distribution")

    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # =========================
    # 🔥 CORRELATION HEATMAP
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(filtered_df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


     # =========================
    # ⚡ SMART INSIGHTS (MOVE HERE)
    # =========================
    st.subheader("⚡ Smart Insights")

    filtered_df = filtered_df.copy()

    if len(filtered_df) > 0:

        insights = []

        churn_rate = filtered_df["Exited"].mean() * 100
        insights.append(f"Churn rate is **{churn_rate:.2f}%** in the selected data.")

        geo_churn = filtered_df.groupby("Geography")["Exited"].mean()
        insights.append(f"Customers in **{geo_churn.idxmax()}** have the highest churn rate.")

        gender_churn = filtered_df.groupby("Gender")["Exited"].mean()
        insights.append(f"**{gender_churn.idxmax()} customers** are more likely to churn.")

        bins = [18, 30, 42, 54, 65, 100]
        labels = ["18-29", "30-41", "42-53", "54-65", "65+"]
        filtered_df["AgeGroup"] = pd.cut(filtered_df["Age"], bins=bins, labels=labels)

        age_churn = filtered_df.groupby("AgeGroup")["Exited"].mean()
        insights.append(f"Customers aged **{age_churn.idxmax()}** have the highest churn risk.")

        for insight in insights:
            st.info(insight)

    else:
        st.warning("No data available for selected filters.")