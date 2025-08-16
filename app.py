# app.py
# Customer Segmentation App using K-Means, PCA, and t-SNE
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ==============================
# Title
# ==============================
st.title("🛍️ Mall Customer Segmentation App")
st.markdown("Cluster mall customers based on spending habits and demographics using **Unsupervised Learning (K-Means)**.")

# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader("Upload the Mall Customers CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Encode Gender
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(int)

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].copy()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================
    # Sidebar Options
    # ==============================
    st.sidebar.header("⚙️ Settings")
    k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 5)
    reduce_method = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])

    # ==============================
    # KMeans Clustering
    # ==============================
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = labels

    # ==============================
    # Evaluation
    # ==============================
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    st.sidebar.markdown(f"**Inertia:** {inertia:.2f}")
    st.sidebar.markdown(f"**Silhouette Score:** {silhouette:.3f}")

    # ==============================
    # Cluster Visualization
    # ==============================
    st.subheader("📉 Cluster Visualization")

    if reduce_method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(X_scaled)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate="auto", init="pca")
        reduced = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    for cluster in range(k):
        ax.scatter(reduced[labels == cluster, 0],
                   reduced[labels == cluster, 1],
                   label=f"Cluster {cluster}")
    ax.set_title(f"Clusters using {reduce_method}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    st.pyplot(fig)

    # ==============================
    # Cluster Profiles
    # ==============================
    st.subheader("📋 Cluster Profiles")
    profile = df.groupby("Cluster")[features].agg(["mean", "median", "min", "max", "count"]).round(2)
    st.dataframe(profile)

    # ==============================
    # Marketing Strategies
    # ==============================
    st.subheader("🎯 Marketing Strategies")

    def describe_segment(row):
        age_mean = row[("Age", "mean")]
        inc_mean = row[("Annual Income (k$)", "mean")]
        spend_mean = row[("Spending Score (1-100)", "mean")]
        traits = []
        if age_mean < 30: traits.append("younger")
        elif age_mean < 45: traits.append("middle-aged")
        else: traits.append("older")
        if inc_mean < 40: traits.append("lower-to-mid income")
        elif inc_mean < 80: traits.append("mid-to-high income")
        else: traits.append("high income")
        if spend_mean < 40: spend_tier = "low spenders"
        elif spend_mean < 70: spend_tier = "moderate spenders"
        else: spend_tier = "high spenders"
        return traits, spend_tier

    def marketing_strategy(row):
        traits, spend_tier = describe_segment(row)
        tactics = []
        if spend_tier == "high spenders":
            tactics += ["VIP rewards", "Exclusive events", "Personalized recommendations"]
        elif spend_tier == "moderate spenders":
            tactics += ["Targeted bundles", "Cross-selling", "Loyalty points"]
        else:
            tactics += ["Discounts", "Awareness campaigns", "Free shipping thresholds"]

        if "younger" in traits:
            tactics += ["Mobile-first campaigns", "Influencer marketing"]
        elif "middle-aged" in traits:
            tactics += ["Family bundles", "Convenience messaging"]
        else:
            tactics += ["Quality focus", "In-store assistance"]

        if "high income" in traits:
            tactics += ["Premium lines", "Exclusive memberships"]
        elif "lower-to-mid income" in traits:
            tactics += ["Value packs", "Installment plans"]

        return traits, spend_tier, tactics[:5]

    strategies = []
    for c, row in profile.iterrows():
        traits, spend_tier, tactics = marketing_strategy(row)
        strategies.append({
            "Cluster": c,
            "Segment Traits": ", ".join(traits) + f"; {spend_tier}",
            "Top 5 Strategies": tactics
        })

    st.dataframe(pd.DataFrame(strategies))

else:
    st.info("Please upload the **Mall_Customers.csv** file to start segmentation.")
