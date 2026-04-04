import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ecommerce ML App", layout="wide")

st.title("🛍️ Ecommerce Customer Analysis App")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Ecommerce CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv('Ecommerce Customers.csv')

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # -----------------------------
    # Basic Info
    # -----------------------------
    st.subheader("📈 Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    with col2:
        st.write("Statistics:")
        st.dataframe(df.describe())

    # -----------------------------
    # EDA Section
    # -----------------------------
    st.subheader("📊 Exploratory Data Analysis")

    if 'Time on Website' in df.columns and 'Yearly Amount Spent' in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df['Time on Website'], df['Yearly Amount Spent'])
        ax.set_title("Time on Website vs Spending")
        st.pyplot(fig)

    if 'Time on App' in df.columns:
        fig, ax = plt.subplots()
        ax.scatter(df['Time on App'], df['Yearly Amount Spent'])
        ax.set_title("Time on App vs Spending")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.hist(df['Yearly Amount Spent'], bins=30)
    ax.set_title("Spending Distribution")
    st.pyplot(fig)

    st.write("Pairplot (sampled for performance)")
    sns_plot = sns.pairplot(df.sample(min(200, len(df))))
    st.pyplot(sns_plot)

    # -----------------------------
    # Encoding
    # -----------------------------
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)

    # -----------------------------
    # KMeans Clustering
    # -----------------------------
    st.subheader("🔍 KMeans Clustering")

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    k = st.slider("Select number of clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_encoded['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("Clustered Data")
    st.dataframe(df_encoded.head())

    # -----------------------------
    # Regression Model
    # -----------------------------
    st.subheader("📈 Linear Regression")

    X_reg = df_encoded.drop(['Yearly Amount Spent', 'Cluster'], axis=1)
    y_reg = df_encoded['Yearly Amount Spent']

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    st.write("R² Score:", lr.score(X_test, y_test))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))

    # -----------------------------
    # Classification Models
    # -----------------------------
    st.subheader("🤖 Classification (Cluster Prediction)")

    X_cls = df_encoded.drop(['Cluster'], axis=1)
    y_cls = df_encoded['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    dt_acc = accuracy_score(y_test, pred_dt)
    rf_acc = accuracy_score(y_test, pred_rf)

    st.write("Decision Tree Accuracy:", dt_acc)
    st.write("Random Forest Accuracy:", rf_acc)

    # -----------------------------
    # Model Comparison
    # -----------------------------
    fig, ax = plt.subplots()
    ax.bar(['Decision Tree', 'Random Forest'], [dt_acc, rf_acc])
    ax.set_title("Model Comparison")
    st.pyplot(fig)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("📊 Feature Importance (Random Forest)")

    importances = rf.feature_importances_
    features = X_cls.columns

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # -----------------------------
    # Tree Visualization
    # -----------------------------
    st.subheader("🌳 Decision Tree Visualization")

    fig = plt.figure(figsize=(20,10))
    plot_tree(dt, filled=True, feature_names=X_cls.columns)
    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to get started")
