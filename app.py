import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

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

# -------------------------------
# Custom Background & CSS
# -------------------------------
page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
}
div[data-testid="stExpander"] {
    background-color: rgba(255,255,255,0.9);
    border-radius: 12px;
    padding: 10px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}
.stTabs [role="tablist"] button {
    font-weight: bold;
    border-radius: 8px;
    padding: 8px;
}
h1 {
    color: #00796b !important;
    text-shadow: 1px 1px 2px #ccc;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
}
@media (max-width: 768px) {
    h1 {
        font-size: 24px !important;
    }
    .stTabs [role="tablist"] button {
        font-size: 12px !important;
        padding: 6px;
    }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🛍️ Customer Segmentation Dashboard")
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

# Select numeric features
X = df.select_dtypes(include=["int64", "float64"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("⚙️ Clustering Options")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
features = st.sidebar.multiselect("Select features for clustering", X.columns.tolist(), default=X.columns.tolist())

if features:
    X_scaled = scaler.fit_transform(df[features])

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA
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
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Shape:", df.shape)

    # Download buttons
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download as CSV", csv, "segmented_customers.csv", "text/csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Customers")
    st.download_button("📊 Download as Excel", buffer.getvalue(), "segmented_customers.xlsx", "application/vnd.ms-excel")

# --- EDA Tab ---
with tab2:
    st.subheader("🔍 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with col2:
        st.write("Feature Distributions")
        for feature in features[:3]:  # show up to 3 features for readability
            fig = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}", color_discrete_sequence=["#00796b"])
            st.plotly_chart(fig, use_container_width=True)

# --- Clustering Tab ---
with tab3:
    st.subheader("📈 Cluster Visualization (PCA 2D)")
    fig = px.scatter(df, x="PCA1", y="PCA2", color="Cluster",
                     title="Customer Clusters (PCA Projection)",
                     hover_data=df.columns,
                     color_continuous_scale=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Cluster Counts")
    fig = px.bar(df["Cluster"].value_counts().reset_index(),
                 x="index", y="Cluster",
                 labels={"index": "Cluster", "Cluster": "Count"},
                 color="index")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Cluster Profiles (Means)")
    st.dataframe(df.groupby("Cluster").mean(numeric_only=True), use_container_width=True)

# --- Insights Tab ---
with tab4:
    st.subheader("💡 Marketing Strategies per Cluster")
    for cluster_id, profile in df.groupby("Cluster").mean(numeric_only=True).iterrows():
        with st.expander(f"📌 Cluster {cluster_id} Strategy"):
            st.write(profile.to_frame().T)

            if "Spending Score (1-100)" in profile.index:
                score = profile["Spending Score (1-100)"]
                if score > 60:
                    st.success("🟢 **High spenders** — Premium offers, loyalty rewards, exclusive deals.")
                elif score < 40:
                    st.warning("🔵 **Low spenders** — Discounts, awareness campaigns, personalized offers.")
                else:
                    st.info("🟡 **Moderate spenders** — Balanced offers & engagement.")
            else:
                st.info("ℹ️ No Spending Score column found — general strategy required.")
