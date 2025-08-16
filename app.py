import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load default dataset
# -------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.markdown("Segment mall customers using **K-Means Clustering** and explore tailored marketing strategies.")

# -------------------------------
# Upload or use default dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    df = load_default_data()
    st.info("ℹ️ Using default **Mall_Customers.csv** dataset")

# -------------------------------
# Data Preprocessing
# -------------------------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(int)
elif "Genre" in df.columns:
    df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1}).astype(int)

# Select only numeric features
X = df.select_dtypes(include=["int64", "float64"])

# Handle scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("⚙️ Clustering Options")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
features = st.sidebar.multiselect("Select features for clustering", X.columns.tolist(), default=X.columns.tolist())

# If features are selected, subset data
if features:
    X_scaled = scaler.fit_transform(df[features])

# Fit KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# -------------------------------
# Tabs Layout
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🔍 EDA", "📈 Clustering", "💡 Insights"])

# --- Dataset Tab ---
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)

# --- EDA Tab ---
with tab2:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Summary Statistics")
        st.dataframe(df.describe())

    with col2:
        st.write("Feature Distributions")
        fig, ax = plt.subplots(figsize=(8,4))
        df[features].hist(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

# --- Clustering Tab ---
with tab3:
    st.subheader("Cluster Visualization (PCA 2D)")
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="tab10", alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

    st.subheader("Cluster Counts")
    st.bar_chart(df["Cluster"].value_counts())

    st.subheader("Cluster Profiles (Means)")
    st.dataframe(df.groupby("Cluster").mean(numeric_only=True))

# --- Insights Tab ---
with tab4:
    st.subheader("Marketing Strategies per Cluster")

    for cluster_id, profile in df.groupby("Cluster").mean(numeric_only=True).iterrows():
        with st.expander(f"📌 Cluster {cluster_id} Strategy"):
            st.write(profile.to_frame().T)

            if "Spending Score (1-100)" in profile.index:
                score = profile["Spending Score (1-100)"]
                if score > 60:
                    st.success("🟢 **High spenders** — Target with premium offers, loyalty rewards, and exclusive deals.")
                elif score < 40:
                    st.warning("🔵 **Low spenders** — Engage with discounts, awareness campaigns, and personalized offers.")
                else:
                    st.info("🟡 **Moderate spenders** — Focus on retention with balanced offers and engagement.")
            else:
                st.info("ℹ️ No Spending Score column found — general strategy required.")

