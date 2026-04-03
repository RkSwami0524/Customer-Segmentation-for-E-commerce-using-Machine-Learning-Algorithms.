import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------- TITLE ----------------
st.title("Customer Segmentation App")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Ecommerce Customers.csv")
data = data.drop(['Email', 'Address', 'Avatar'], axis=1)

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------- FEATURE SELECTION ----------------
X = data[['Avg. Session Length', 'Time on App', 'Time on Website',
          'Length of Membership', 'Yearly Amount Spent']]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")
k = st.sidebar.slider("Select number of clusters", 2, 10, 4)

# ---------------- KMEANS ----------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans

# ---------------- VISUALIZATION ----------------
st.subheader("Customer Segments")

fig, ax = plt.subplots()
ax.scatter(data['Time on App'], data['Yearly Amount Spent'],
           c=data['Cluster'], cmap='viridis')

ax.set_xlabel("Time on App")
ax.set_ylabel("Yearly Amount Spent")
ax.set_title("Customer Segments")

st.pyplot(fig)

# ---------------- INSIGHTS ----------------
st.subheader("Cluster Insights")
cluster_analysis = data.groupby('Cluster').mean()
st.dataframe(cluster_analysis)

# ---------------- OPTIONAL HEATMAP ----------------
if st.checkbox("Show Correlation Heatmap"):
    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

# ---------------- OPTIONAL PAIRPLOT ----------------
if st.checkbox("Show Pairplot (slow)"):
    pairplot_fig = sns.pairplot(data, hue='Cluster')
    st.pyplot(pairplot_fig)
