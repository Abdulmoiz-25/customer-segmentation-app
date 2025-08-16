import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load default dataset
# -------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")

# Title
st.title("🛍️ Mall Customer Segmentation using K-Means")

# File upload
uploaded_file = st.file_uploader("Upload your own dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    df = load_default_data()
    st.info("ℹ️ Using default Mall_Customers.csv dataset")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(int)
elif "Genre" in df.columns:
    df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1}).astype(int)
else:
    st.warning("⚠️ No 'Gender' or 'Genre' column found. Skipping gender encoding.")

# Select numeric features
X = df.select_dtypes(include=["int64", "float64"])

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sidebar - clustering options
# -------------------------------
st.sidebar.header("Clustering Options")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 5)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# -------------------------------
# Results
# -------------------------------
# Cluster counts
st.subheader("Cluster Counts")
st.write(df["Cluster"].value_counts())

# Plot clusters
st.subheader("Cluster Visualization (PCA 2D)")
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis")
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

# Cluster profiling
st.subheader("Cluster Profiles")
st.write(df.groupby("Cluster").mean(numeric_only=True))

# Marketing strategies suggestion
st.subheader("Suggested Marketing Strategies")
for cluster_id, profile in df.groupby("Cluster").mean(numeric_only=True).iterrows():
    st.markdown(f"### Cluster {cluster_id}")
    if "Spending Score (1-100)" in profile.index and profile["Spending Score (1-100)"] > 60:
        st.write("🟢 High spenders — Target with premium offers, loyalty rewards, and exclusive deals.")
    elif "Spending Score (1-100)" in profile.index and profile["Spending Score (1-100)"] < 40:
        st.write("🔵 Low spenders — Engage with discounts, awareness campaigns, and personalized offers.")
    else:
        st.write("🟡 Moderate spenders — Focus on retention with regular engagement and balanced deals.")
