# Generated from: Ecommerce Customers.ipynb
# Converted at: 2026-04-03T17:20:08.696Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Project Title
# ## Customer Segmentation for E-commerce using Machine Learning Algorithms


# ## Introduction
# E-commerce platforms collect detailed data about customer behavior such as time spent on apps, websites, and purchasing patterns. Analyzing this data helps businesses understand customers better.
# 
# This project uses Machine Learning (K-Means Clustering) to segment customers based on their behavior using the given dataset “Ecommerce Customers.csv”.
# 


# ## Dataset Description
# 
# The dataset contains the following important features:
# 
# - Avg. Session Length
# - Time on App
# - Time on Website
# - Length of Membership
# - Yearly Amount Spent
# 
# These features represent customer engagement and spending behavior.


# ## Objective
# - Segment customers into groups
# - Identify high-value and low-value customers
# - Help businesses improve marketing strategies


# ## Technologies Used
# - Python
# - Pandas & NumPy
# - Matplotlib & Seaborn
# - Scikit-learn
# - Jupyter Notebook


# ## Implementation with Code & Explanation
# ## Import Libraries
# 
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ## Explanation
# - pandas → used for handling data tables
# - numpy → used for numerical calculations
# - matplotlib & seaborn → used for visualization
# - StandardScaler → normalizes data
# - KMeans → clustering algorithm
# - silhouette_score → evaluates clustering
# 


# ## Load Dataset
# 


data = pd.read_csv("Ecommerce Customers.csv")
print(data.head())

# ## Explanation
# - read_csv() loads dataset
# - head() shows first 5 rows
# - Helps understand structure of data


# ## Data Preprocessing


print(data.info())

# ## Explanation
# - Shows dataset structure
# - Checks data types


print(data.isnull().sum())

# ## Explanation
# - Checks missing values


data = data.drop(['Email', 'Address', 'Avatar'], axis=1)

# ## Explanation
# - Removes unnecessary columns
# - These do not affect clustering


# ## Exploratory Data Analysis (EDA)


sns.pairplot(data)
plt.show()

# ## Explanation
# - Shows relationships between features
# - Helps understand patterns


sns.heatmap(data.corr(), annot=True)
plt.show()

# ## Explanation
# - Displays correlation between variables
# - Helps identify important features


# ## Feature Selection


X = data[['Avg. Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent']]

# ## Explanation
# - Selects important behavioral features
# - These define customer activity


# ## Feature Scaling


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ## Explanation
# - Normalizes data
# - Ensures all features have equal importance


# ## Elbow Method


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# ## Explanation
# - Tests cluster values from 1 to 10
# - Stores WCSS (error) values


plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ## Explanation
# - Graph helps find optimal clusters
# - Elbow point = best cluster number


# ## Apply K-Means


kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# ## Explanation
# - Chooses 4 clusters (example from elbow)
# - Assigns cluster to each customer


data['Cluster'] = y_kmeans

# ## Explanation
# - Adds cluster labels to dataset


# ## Evaluate Model


score = silhouette_score(X_scaled, y_kmeans)
print("Silhouette Score:", score)

# ## Explanation
# - Measures clustering performance
# - Higher score = better segmentation


# ## Visualization


plt.figure(figsize=(8,6))
plt.scatter(data['Time on App'], data['Yearly Amount Spent'], 
            c=data['Cluster'], cmap='viridis')

plt.xlabel("Time on App")
plt.ylabel("Yearly Amount Spent")
plt.title("Customer Segments")
plt.show()

# ## Explanation
# - Visualizes customer groups
# - Different colors = different segments


# ## Cluster Analysis


cluster_analysis = data.groupby('Cluster').mean()
print(cluster_analysis)

# ## Explanation
# - Calculates average values per cluster
# - Helps understand each segment


# ## Result Interpretation
# 
# Example clusters:
# 
# - Cluster 0 → High spending, high app usage
# - Cluster 1 → Low spending customers
# - Cluster 2 → Moderate users
# - Cluster 3 → Loyal long-term customers


# ## Applications
# - Targeted advertisements
# - Personalized recommendations
# - Customer retention strategies


# ## Conclusion
# 
# This project successfully segments customers using machine learning. Businesses can use these insights to improve marketing strategies and enhance customer satisfaction.